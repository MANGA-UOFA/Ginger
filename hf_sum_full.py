import logging
import math
import os
import time
from pathlib import Path
from typing import Callable

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import datasets
import evaluate
import hydra
import jax
import jax.numpy as jnp
import nltk
import numpy as np
import optax
import transformers
import wandb
from datasets import Dataset, load_dataset
from filelock import FileLock
from flax import traverse_util
from flax.training import train_state
from flax.training.common_utils import onehot
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    is_tensorboard_available,
    set_seed,
)
from transformers.utils import is_offline_mode

import optimizers

logger = logging.getLogger(__name__)
datasets.enable_caching()

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray


def state_update(state, grads, ngrads, lr, weight_decay, **kwargs):
    updates, new_opt_state = state.tx.update(grads, state.opt_state, ngrads)
    new_params = optax.apply_updates(state.params, updates)
    new_params = jax.tree_map(lambda x, y: x - weight_decay * lr * y, new_params, state.params)
    return (
        state.replace(
            step=state.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        ),
        updates,
    )


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False, drop_last=True):
    """
    Returns batches of size `batch_size` from `dataset`. If `drop_last` is set to `False`, the final batch may be incomplete, and range in size from 1 to `batch_size`. Shuffle batches if `shuffle` is `True`.
    """  # noqa
    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
        batch_idx = np.asarray(batch_idx)
    else:
        batch_idx = np.arange(len(dataset))

    if drop_last:
        steps_per_epoch = len(dataset) // batch_size
        batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
        batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))
    else:
        steps_per_epoch = math.ceil(len(dataset) / batch_size)
        batch_idx = np.array_split(batch_idx, steps_per_epoch)

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: np.array(v) for k, v in batch.items()}

        yield batch


def write_train_metric(summary_writer, train_metrics, train_time, step):
    length = len(train_metrics)
    for i, metric in enumerate(train_metrics):
        for key, val in metric.items():
            tag = f"train/{key}"
            summary_writer.scalar(tag, val, step - length + i + 1)
            wandb.log({tag: val}, step=step - length + i + 1)
    summary_writer.scalar("train/time", train_time, step)
    wandb.log({"train/time": train_time}, step=step)


def write_eval_metric(summary_writer, eval_metrics, step):
    for metric_name, value in eval_metrics.items():
        tag = f"eval/{metric_name}"
        summary_writer.scalar(tag, value, step)
        wandb.log({tag: value}, step=step)


def create_learning_rate_fn(
    train_ds_size: int,
    train_batch_size: int,
    num_train_epochs: int,
    num_warmup_steps: int,
    learning_rate: float,
    lr_decay: bool = True,
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate,
        end_value=0.0 if lr_decay else learning_rate,
        transition_steps=num_train_steps - num_warmup_steps,
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def main(cfg):
    if (
        os.path.exists(cfg.training.output_dir)
        and os.listdir(cfg.training.output_dir)
        and cfg.training.do_train
        and not cfg.training.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({cfg.training.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {cfg.training}")

    # Set seed before initializing model.
    set_seed(cfg.training.seed)

    if cfg.data.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config_name,
            cache_dir=cfg.model.cache_dir,
            keep_in_memory=False,
            use_auth_token=True if cfg.model.use_auth_token else None,
        )
    else:
        raise ValueError("Need a dataset name")
    # Load pretrained model and tokenizer

    if cfg.model.config_name:
        config = AutoConfig.from_pretrained(
            cfg.model.config_name,
            cache_dir=cfg.model.cache_dir,
        )
    elif cfg.model.model_name_or_path:
        config = AutoConfig.from_pretrained(
            cfg.model.model_name_or_path,
            cache_dir=cfg.model.cache_dir,
        )
    else:
        config = CONFIG_MAPPING[cfg.model.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if cfg.model.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.tokenizer_name,
            cache_dir=cfg.model.cache_dir,
            use_fast=cfg.model.use_fast_tokenizer,
            use_auth_token=True if cfg.model.use_auth_token else None,
        )
    elif cfg.model.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.model_name_or_path,
            cache_dir=cfg.model.cache_dir,
            use_fast=cfg.model.use_fast_tokenizer,
            use_auth_token=True if cfg.model.use_auth_token else None,
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if cfg.model.model_name_or_path:
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            cfg.model.model_name_or_path,
            config=config,
            seed=cfg.training.seed,
            dtype=getattr(jnp, cfg.model.dtype),
            use_auth_token=True if cfg.model.use_auth_token else None,
        )
    else:
        model = FlaxAutoModelForSeq2SeqLM.from_config(
            config,
            seed=cfg.training.seed,
            dtype=getattr(jnp, cfg.model.dtype),
        )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = cfg.data.source_prefix if cfg.data.source_prefix is not None else ""
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if cfg.training.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        column_names = dataset["train"].column_names
    elif cfg.training.do_eval:
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        column_names = dataset["validation"].column_names
    elif cfg.training.do_predict:
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(cfg.data.dataset_name, None)
    if cfg.data.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = cfg.data.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{cfg.data.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if cfg.data.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = cfg.data.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{cfg.data.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = cfg.data.max_target_length

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=cfg.data.max_source_length, padding="max_length", truncation=True, return_tensors="np"
        )

        # Setup the tokenizer for targets
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right_fn(
            labels["input_ids"], config.pad_token_id, config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

        # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    preprocess_function(dataset["train"][:1])  # dummy run

    if cfg.training.do_train:
        train_dataset = dataset["train"]
        if cfg.data.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), cfg.data.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if cfg.training.do_eval:
        max_target_length = cfg.data.val_max_target_length
        eval_dataset = dataset["validation"]
        if cfg.data.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), cfg.data.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if cfg.training.do_predict:
        max_target_length = cfg.data.val_max_target_length
        predict_dataset = dataset["test"]
        if cfg.data.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), cfg.data.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=cfg.data.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not cfg.data.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )

    # Metric
    metric = evaluate.load("rouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(preds, labels):
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Enable tensorboard only on the master node
    has_tensorboard = is_tensorboard_available()
    if has_tensorboard and jax.process_index() == 0:
        name = f"{cfg.optimizer.name}-lr({cfg.optimizer.learning_rate})-wd({cfg.training.weight_decay}){ f'tau({cfg.optimizer.tau})' if hasattr(cfg.optimizer, 'tau') else ''}"  # noqa: E501
        try:
            wandb.init(
                project="summarization",
                config=dict(OmegaConf.to_container(cfg, resolve=True)),
                # sync_tensorboard=True,
                dir=f"{cfg.training.output_dir}/",
                name=name,
            )
            wandb.define_metric("eval/loss", summary="min")
            wandb.define_metric("eval/rouge1", summary="max")
            wandb.define_metric("train/loss", summary="min")

            name = name + os.environ.get("SLURM_JOB_ID", wandb.run.id)
            from flax.metrics import tensorboard as tb

            summary_writer = tb.SummaryWriter(log_dir=Path(cfg.training.output_dir) / name)
            summary_writer.hparams(dict(training=vars(cfg.training), data=vars(cfg.data), model=vars(cfg.model)))

        except ImportError as ie:
            has_tensorboard = False
            logger.warning(
                f"Unable to display metrics through TensorBoard because some package are not installed: {ie}"
            )
    else:
        logger.warning(
            "Unable to display metrics through TensorBoard because the package is not installed: "
            "Please run pip install tensorboard to enable."
        )

    # Initialize our training
    rng = jax.random.PRNGKey(cfg.training.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(cfg.training.num_train_epochs)
    train_batch_size = (
        int(cfg.training.per_device_train_batch_size) * jax.device_count() * cfg.training.gradient_accumulation_steps
    )
    per_device_eval_batch_size = int(cfg.training.per_device_eval_batch_size)
    eval_batch_size = per_device_eval_batch_size * jax.device_count()
    steps_per_epoch = len(train_dataset) // train_batch_size
    total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        # find out all LayerNorm parameters
        layer_norm_candidates = ["layernorm", "layer_norm", "ln"]
        layer_norm_named_params = {
            layer[-2:]
            for layer_norm_name in layer_norm_candidates
            for layer in flat_params.keys()
            if layer_norm_name in "".join(layer).lower()
        }
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_named_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    optimizer_cfg = OmegaConf.to_object(cfg.optimizer)
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        len(train_dataset),
        train_batch_size,
        cfg.training.num_train_epochs,
        cfg.training.warmup_steps,
        optimizer_cfg.pop("learning_rate"),
        lr_decay=cfg.training.lr_decay,
    )
    optimizer_name = optimizer_cfg.pop("name")
    optimizer_cls = optimizers.get_optimizer_cls(optimizer_name)
    optimizer = optimizer_cls(learning_rate=linear_decay_lr_schedule_fn, **optimizer_cfg)

    optimizer = optax.chain(
        optax.clip_by_global_norm(cfg.training.max_grad_norm),
        optimizer,
    )

    # Setup train state
    state = TrainState.create(apply_fn=model.__call__, params=model.params, tx=optimizer, dropout_rng=dropout_rng)

    # label smoothed cross entropy
    def loss_fn(logits, labels, padding_mask, label_smoothing_factor=0.0):
        """
        The label smoothing implementation is adapted from Flax's official example:
        https://github.com/google/flax/blob/87a211135c6a377c8f29048a1cac3840e38b9da4/examples/wmt/train.py#L104
        """
        vocab_size = logits.shape[-1]
        confidence = 1.0 - label_smoothing_factor
        low_confidence = (1.0 - confidence) / (vocab_size - 1)
        normalizing_constant = -(
            confidence * jnp.log(confidence) + (vocab_size - 1) * low_confidence * jnp.log(low_confidence + 1e-20)
        )
        soft_labels = onehot(labels, vocab_size, on_value=confidence, off_value=low_confidence)

        loss = optax.softmax_cross_entropy(logits, soft_labels)
        loss = loss - normalizing_constant

        # ignore padded tokens from loss
        loss = loss * padding_mask
        loss = loss.sum()
        num_labels = padding_mask.sum()
        return loss / num_labels, num_labels

    # Define gradient update step fn
    @jax.jit
    def train_step(state, batch, label_smoothing_factor=0.0):
        def compute_loss(params, inputs, rng):
            labels = inputs.pop("labels")
            logits = state.apply_fn(**inputs, params=params, dropout_rng=rng, train=True)[0]
            loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
            return loss, num_labels

        def compute_natural_loss(params, inputs, rng):
            rng, pred_rng = jax.random.split(rng)
            # _ = inputs.pop("labels")
            logits = state.apply_fn(**inputs, params=params, dropout_rng=rng, train=True)[0]
            labels = jax.random.categorical(pred_rng, logits, -1)
            loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)
            return loss

        def scan_body(carry, step_inputs):
            loss, num_labels, grad, ngrad, rng = carry

            grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
            ngrad_fn = jax.grad(compute_natural_loss)
            (step_loss, step_num_labels), step_grad = grad_fn(state.params, step_inputs, rng)
            step_ngrad = (
                ngrad_fn(state.params, step_inputs, rng)
                if (("sgd" not in cfg.optimizer.name) or ("adam" not in cfg.optimizer.name))
                else step_grad
            )
            step_ngrad = jax.tree_map(lambda x: jnp.clip(x, -1e10, 1e10), step_ngrad)

            def rescale(x, step_x):
                return (x * num_labels + step_x * step_num_labels) / (num_labels + step_num_labels)

            loss = rescale(loss, step_loss)
            grad = jax.tree_map(lambda x, y: rescale(x, y), grad, step_grad)
            ngrad = jax.tree_map(lambda x, y: rescale(x, y), ngrad, step_ngrad)
            rng = jax.random.split(rng)[-1]

            return (loss, num_labels + step_num_labels, grad, ngrad, rng), None

        (loss, num_labels, grad, ngrad, new_dropout_rng), _ = jax.lax.scan(
            scan_body,
            (
                0.0,
                0,
                jax.tree_map(jnp.zeros_like, state.params),
                jax.tree_map(jnp.zeros_like, state.params),
                state.dropout_rng,
            ),
            batch,
        )

        ngrad = jax.tree_map(lambda x: x * jnp.sqrt(num_labels), ngrad)

        new_state, updates = state_update(
            state,
            grad,
            ngrad,
            linear_decay_lr_schedule_fn(state.step),
            cfg.training.weight_decay,
            dropout_rng=new_dropout_rng,
        )

        _updates = jax.flatten_util.ravel_pytree(updates)[0]
        _grad = jax.flatten_util.ravel_pytree(grad)[0]
        _ngrad = jax.flatten_util.ravel_pytree(ngrad)[0]

        metrics = {
            "loss": loss,
            "learning_rate": linear_decay_lr_schedule_fn(new_state.step),
            "updates_norm": jnp.linalg.norm(_updates),
            "grad_norm": jnp.linalg.norm(_grad),
            "ngrad_norm": jnp.linalg.norm(_ngrad),
            "cosine": jnp.dot(_updates, _grad) / (jnp.linalg.norm(_updates) * jnp.linalg.norm(_grad)),
            "%truncated": jnp.sum(jnp.abs(_updates) == jnp.abs(_updates).max()) / _updates.size,
        }

        return new_state, metrics

    # Define eval fn
    @jax.jit
    def eval_step(state, batch, label_smoothing_factor=0.0):
        labels = batch.pop("labels")
        logits = model(**batch, params=state.params, train=False)[0]
        loss, num_labels = loss_fn(logits, labels, batch["decoder_attention_mask"], label_smoothing_factor)

        metrics = {"loss": loss}
        return metrics

    # Define generation function
    max_length = (
        cfg.data.val_max_target_length if cfg.data.val_max_target_length is not None else model.config.max_length
    )
    num_beams = cfg.data.num_beams if cfg.data.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    @jax.jit
    def generate_step(state, batch):
        output_ids = model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            params=state.params,
            **gen_kwargs,
        )
        return output_ids.sequences

    # Create parallel version of the train and eval step

    # Replicate the train state on each device

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.training.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel & distributed) = {train_batch_size}")
    logger.info(f"  Total optimization steps = {total_train_steps}")

    train_start = time.time()
    train_metrics = []
    epochs = tqdm(range(num_epochs), desc="Epoch ... ", position=0)
    for epoch in epochs:
        # ======================== Training ================================

        # Create sampling rng
        rng, input_rng = jax.random.split(rng)

        # Generate an epoch by shuffling sampling indices from the train dataset
        train_loader = data_loader(input_rng, train_dataset, train_batch_size, shuffle=True)
        steps_per_epoch = len(train_dataset) // train_batch_size
        # train
        for step in tqdm(range(steps_per_epoch), desc="Training...", position=1, leave=False):
            batch = next(train_loader)  # (#dev * #gacc * bsz, seqlen)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((-1, cfg.training.per_device_train_batch_size) + x.shape[1:]), batch
            )  # (#dev * #gacc, bsz, seqlen)

            state, train_metric = train_step(state, batch, cfg.training.label_smoothing_factor)
            train_metrics.append(train_metric)

            cur_step = epoch * (len(train_dataset) // train_batch_size) + step
            if cur_step % cfg.training.logging_steps == 0 and cur_step > 0:
                train_time = time.time() - train_start
                if has_tensorboard and jax.process_index() == 0:
                    write_train_metric(summary_writer, train_metrics, train_time, cur_step)
                    train_metrics = []

            if (cur_step % cfg.training.eval_steps == 0 or cur_step + 1 == total_train_steps) and cur_step > 0:
                # ======================== Evaluating ==============================
                eval_metrics = []
                eval_preds = []
                eval_labels = []
                eval_loader = data_loader(input_rng, eval_dataset, eval_batch_size, drop_last=False)
                eval_steps = math.ceil(len(eval_dataset) / eval_batch_size)
                for _ in tqdm(range(eval_steps), desc="Evaluating...", position=2, leave=False):
                    # Model forward
                    batch = next(eval_loader)
                    labels = batch["labels"]

                    metrics = eval_step(state, batch, cfg.training.label_smoothing_factor)
                    eval_metrics.append(metrics)

                    # generation
                    if cfg.data.predict_with_generate:
                        generated_ids = generate_step(state, batch)
                        eval_preds.extend(jax.device_get(generated_ids))
                        eval_labels.extend(labels)

                eval_keys = eval_metrics[0].keys()
                eval_metrics = {k: jnp.stack([metrics[k] for metrics in eval_metrics]).mean() for k in eval_keys}

                # compute ROUGE metrics
                if cfg.data.predict_with_generate:
                    rouge_metrics = compute_metrics(eval_preds, eval_labels)
                    eval_metrics.update(rouge_metrics)

                # Print metrics and update progress bar
                desc = f"Step... ({cur_step} | Eval Loss: {eval_metrics['loss']}"
                epochs.write(desc)
                epochs.desc = desc

                # Save metrics
                if has_tensorboard and jax.process_index() == 0:
                    write_eval_metric(summary_writer, eval_metrics, cur_step)


@hydra.main(version_base=None, config_path="configs", config_name="sum_full")
def launch(cfg: OmegaConf) -> None:
    if cfg.training.output_dir is not None:
        cfg.training.output_dir = os.path.expanduser(cfg.training.output_dir)

    if cfg.data.dataset_name is None and cfg.data.train_file is None and cfg.data.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if cfg.data.train_file is not None:
            extension = cfg.data.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("train_file` should be a csv, json or text file.")
        if cfg.data.validation_file is not None:
            extension = cfg.data.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt"]:
                raise ValueError("`validation_file` should be a csv, json or text file.")
    if cfg.data.val_max_target_length is None:
        cfg.data.val_max_target_length = cfg.data.max_target_length

    main(cfg)


if __name__ == "__main__":
    launch()
