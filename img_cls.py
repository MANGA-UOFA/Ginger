import multiprocessing as mp
import os
import random
import time

import datasets
import hydra
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from flax.training import early_stopping
from omegaconf import OmegaConf

# from torchvision import transforms
from PIL import Image

from models import get_model
from optimizers import get_optimizer_cls


os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
datasets.enable_caching()


class Tracker:
    def __init__(self, config) -> None:
        self.config = config
        run_name = f"{config.optimizer}-d{config.weight_decay}-sd{config.int_weight_decay}"
        wandb.init(
            project=f"ENG-{config.model.name}-{config.dataset.name}",
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
            dir=".logs/",
        )
        wandb.define_metric("valid/acc", summary="max")
        wandb.define_metric("valid/loss", summary="min")
        wandb.define_metric("train/loss", summary="min")
        self.run = wandb.run

    def log(self, info_dict, step=None):
        wandb.log(info_dict, step=step)


def seed_rngs(seed):
    np.random.seed(seed)
    random.seed(seed)


def get_dataset_mean_std(name=None):
    mean = [0.5, 0.5, 0.5]
    std = [0.2, 0.2, 0.2]
    if name is None:
        return mean, std
    if name == "cifar100":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    return mean, std


def get_dataset_mapping(name):
    if name == "cifar100":
        dsmp = {
            "image": "img",
            "label": "fine_label",
        }
    elif name == "cifar10":
        dsmp = {
            "image": "img",
            "label": "label",
        }
    else:
        dsmp = {
            "image": "image",
            "label": "label",
        }
    return dsmp


def __transform(images, name=None, rng=None):
    images = np.asarray(images) / 255.0
    mean, std = get_dataset_mean_std(name)
    mean = np.asarray(mean)
    std = np.asarray(std)
    images = (images - mean) / std
    return images


def __augment(image, nprng):
    crop_x, crop_y, flip = nprng.random_raw(3)
    crop_x = crop_x.item() % 8
    crop_y = crop_y.item() % 8
    flip = flip.item() % 2

    def random_crop(image):
        image = np.pad(image, ((4, 4), (4, 4), (0, 0)), mode="reflect")
        image = image[crop_x : crop_x + 32, crop_y : crop_y + 32, :]
        return image

    def random_flip(image):
        if flip == 0:
            image = np.flip(image, axis=1)
        return image

    return Image.fromarray(random_flip(random_crop(np.asarray(image))))


def augment(x, idx, nprng, augmentation):
    return {
        "image": __augment(x["image"], nprng.advance(idx)) if augmentation else x["image"],
    }


def preprocess(dataset, name):
    dsmp = get_dataset_mapping(name)

    return dataset.map(
        lambda x: {
            "image": x[dsmp["image"]].convert("RGB"),
            "label": x[dsmp["label"]],
        },
        remove_columns=dataset.column_names,
        num_proc=mp.cpu_count(),
    )


def acc_fn(model, params, data):
    images = data["image"]
    labels = data["label"]
    pred = model.apply(params, images, train=False)
    loss = -jnp.take_along_axis(jax.nn.log_softmax(pred, -1), labels[..., None], -1)
    acc = jnp.equal(jnp.argmax(pred, -1), labels)
    return loss.mean(dtype=jnp.float32), acc.mean(dtype=jnp.float32)


def loss_fn(rng, model, params, images, labels):
    rng, _ = jax.random.split(rng)
    pred, mutated_vars = model.apply(params, images, train=True, mutable=["batch_stats"])
    return (
        -jnp.take_along_axis(jax.nn.log_softmax(pred, -1), labels[..., None], -1).mean(),
        mutated_vars["batch_stats"],
    )


def natural_loss_fn(model, params, rng, images, labels):
    pred = model.apply(params, images, train=False)
    labels = jax.random.categorical(rng, pred, -1)
    return -jnp.take_along_axis(jax.nn.log_softmax(pred, -1), labels[..., None], -1).mean()


def train(config):
    seed_rngs(config.seed)
    rng = jax.random.PRNGKey(config.seed)

    logs = {
        "wall": [],
        "loss": [],
        "valid": {"acc": [], "wall": [], "loss": []},
    }
    steps = 0

    dsmp = get_dataset_mapping(config.dataset.name)
    dataset = datasets.load_dataset(config.dataset.name, split=config.dataset.train)
    num_classes = dataset.features[dsmp["label"]].num_classes
    dataset = preprocess(dataset, config.dataset.name)
    valid_ds = datasets.load_dataset(config.dataset.name, split=config.dataset.valid)
    valid_ds = preprocess(valid_ds, config.dataset.name)
    valid_ds = valid_ds.with_format("numpy")
    es = early_stopping.EarlyStopping(patience=config.patience)

    model_config = OmegaConf.to_object(config.model)
    model_config["num_classes"] = num_classes
    model = get_model(**model_config)
    dummy_image = jnp.ones((config.batch_size, 32, 32, 3))
    variables = model.init(rng, dummy_image)
    epoch_length = (len(dataset) + config.batch_size - 1) // config.batch_size
    optimizer_config = OmegaConf.to_object(config.optimizer)
    optimizer_cls = get_optimizer_cls(optimizer_config.pop("name"))
    scheduler = optax.warmup_exponential_decay_schedule(
        init_value=1e-7,
        peak_value=optimizer_config.pop("learning_rate"),
        warmup_steps=epoch_length * config.warmup,
        transition_steps=epoch_length * 50,
        decay_rate=config.lr_decay_rate,
        staircase=True,
    )
    optim = optimizer_cls(learning_rate=scheduler, **optimizer_config)
    state = optim.init(variables["params"])
    variables = {
        "params": variables["params"],
        "batch_stats": variables["batch_stats"],
    }

    @jax.jit
    def update_fn(variables, state, batch, rng, steps):
        train_params = jax.tree_map(lambda x: x.astype(config.model.dtype), variables)
        batch = jax.tree_map(
            lambda x: x.reshape(config.gradient_accumulation, -1, *x.shape[1:]),
            batch,
        )

        def _update(carry, _batch):
            rng, acc_grads, acc_ngrads, acc_batch_stats = carry

            rng, subkey = jax.random.split(rng)
            (value, batch_stats), grads = jax.value_and_grad(loss_fn, argnums=2, has_aux=True)(
                subkey,
                model,
                train_params,
                _batch["image"],
                _batch["label"],
            )

            batch_stats = jax.tree_map(lambda x, y: x - y, batch_stats, acc_batch_stats)

            rng, subkey = jax.random.split(rng)
            ngrads = jax.grad(natural_loss_fn, argnums=1)(
                model,
                train_params,
                subkey,
                _batch["image"],
                _batch["label"],
            )

            acc_grads = jax.tree_map(
                lambda x, y: x + y / config.gradient_accumulation,
                acc_grads,
                grads,
            )

            acc_ngrads = jax.tree_map(
                lambda x, y: x + y / config.gradient_accumulation,  # sum over batch
                acc_ngrads,
                ngrads,
            )

            acc_batch_stats = jax.tree_map(
                lambda x, y: x + y / config.gradient_accumulation,
                acc_batch_stats,
                batch_stats,
            )

            return (rng, acc_grads, acc_ngrads, acc_batch_stats), value

        (rng, grads, ngrads, batch_stats), values = jax.lax.scan(
            _update,
            (
                rng,
                jax.tree_map(jnp.zeros_like, train_params),
                jax.tree_map(jnp.zeros_like, train_params),
                jax.tree_map(jnp.zeros_like, variables["batch_stats"]),
            ),
            batch,
        )
        ngrads = jax.tree_map(lambda x: x * jnp.sqrt(config.batch_size), ngrads)
        value = jnp.mean(values, dtype=jnp.float32)

        updates = grads["params"]
        if config.int_weight_decay:
            updates = jax.tree_map(lambda x, y: x + config.weight_decay * y, grads["params"], variables["params"])
        updates, state = optim.update(updates, state, ngrads["params"])
        updates = jax.tree_map(lambda x: jnp.where(jnp.isnan(x), 0.0, x), updates)

        lr = scheduler(steps + 1)
        if not config.int_weight_decay:
            updates = jax.tree_map(lambda x, y: x - lr * config.weight_decay * y, updates, variables["params"])

        variables["params"] = optax.apply_updates(variables["params"], updates)
        variables["batch_stats"] = batch_stats

        fg = jax.flatten_util.ravel_pytree(updates)[0]

        info = {
            "mean": fg.mean(),
            "std": fg.std(),
            "max": fg.max(),
            "min": fg.min(),
            "2norm": jnp.linalg.norm(fg),
            "1norm": jnp.linalg.norm(fg, ord=1),
            "avg2norm": jnp.linalg.norm(fg) / jnp.sqrt(fg.size),
            "avg1norm": jnp.linalg.norm(fg, ord=1) / fg.size,
        }
        return variables, info, state, value, rng

    @jax.jit
    def jitted_acc(variables, data):
        return acc_fn(model, variables, data)

    def validate(variables):
        acc = 0
        loss = 0
        count = 0
        for data in valid_ds.iter(batch_size=config.batch_size):
            data = {
                "image": __transform(data["image"], config.dataset.name),
                "label": np.asarray(data["label"]),
            }

            _loss, _acc = jitted_acc(variables, data)
            loss += _loss * data["image"].shape[0]
            acc += _acc * data["image"].shape[0]
            count += data["image"].shape[0]
        return loss / count, acc / count

    start = time.time()

    tracker = Tracker(config)
    total_iters = epoch_length * config.max_epochs
    bar = tqdm.tqdm(total=total_iters, desc="Training")
    for epoch in range(config.max_epochs):
        shuffle_seed = jax.random.randint(rng, (1,), minval=0, maxval=jnp.iinfo(jnp.int32).max).item()
        _, rng = jax.random.split(rng)
        aug_dataset = (
            dataset.shuffle(seed=shuffle_seed)
            .map(
                augment,
                num_proc=mp.cpu_count(),
                with_indices=True,
                fn_kwargs={"nprng": np.random.PCG64(shuffle_seed), "augmentation": config.augmentation},
            )
            .with_format("numpy")
        )
        for batch in aug_dataset.iter(batch_size=config.batch_size):
            batch = {
                "image": __transform(batch["image"], config.dataset.name),
                "label": np.asarray(batch["label"]),
            }
            if config.dryrun:
                continue
            _, rng = jax.random.split(rng)

            variables, info, state, value, rng = update_fn(variables, state, batch, rng, steps)

            logs["wall"].append(time.time() - start)
            logs["loss"].append(value)

            tracker.log(
                {
                    "train/loss": value,
                    "train/wall": time.time() - start,
                    "train/lr": scheduler(steps),
                },
                step=steps,
            )
            tracker.log(
                {f"train/updates/{k}": v for k, v in info.items()},
                step=steps,
            )

            if steps % config.eval_steps == 0:
                loss, acc = validate(variables)
                loss = jax.device_get(loss)
                acc = jax.device_get(acc)
                logs["valid"]["acc"].append(acc)
                logs["valid"]["wall"].append(time.time() - start)
                logs["valid"]["loss"].append(loss)
                tracker.log(
                    {
                        "valid/acc": acc,
                        "valid/loss": loss,
                    },
                    step=steps,
                )
                improved, es = es.update(-acc)

                def check_nonfinite(pytree):
                    return jax.tree_util.tree_reduce(
                        lambda x, y: jnp.logical_or(x, y), jax.tree_map(lambda x: np.isnan(x).any(), pytree), False
                    )

                if es.should_stop or loss > 1e2 or acc < 0.000001 or check_nonfinite(variables):
                    wandb.finish()
                    return
            steps += 1
            bar.update()
        tracker.log({"train/epoch": epoch}, step=steps)
        tracker.log({"train/wall": time.time() - start}, step=steps)


@hydra.main(version_base=None, config_path="configs", config_name="img-cls")
def launch(config):
    if config.debug.nans:
        jax.config.update("jax_debug_nans", True)
    if config.debug.disable_jit:
        jax.config.update("jax_disable_jit", True)

    train(config)


if __name__ == "__main__":
    launch()
