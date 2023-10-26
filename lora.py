import flax
import jax
import jax.numpy as jnp

LORA_FULL = 0
LORA_FREEZE = -1


def simple_spec(params, decision_fn, tune_vectors):
    d = flax.traverse_util.flatten_dict(params, sep=".")
    spec = {}
    for path, arr in d.items():
        if arr.ndim <= 1:
            spec[path] = LORA_FULL if tune_vectors else LORA_FREEZE
        else:
            spec[path] = decision_fn(path, arr)
    return spec


def init_lora(params, lora_spec, rng, std=0.01, alpha=32):
    trainable = {}
    freezed = {}
    for path, arr in flax.traverse_util.flatten_dict(params, sep=".").items():
        if lora_spec[path] == LORA_FULL:
            trainable[path] = arr
        elif lora_spec[path] == LORA_FREEZE:
            freezed[path] = arr
        else:
            trainable[f"{path}.lora.a"] = jax.random.normal(rng, (arr.shape[0], lora_spec[path]), dtype=arr.dtype) * std
            trainable[f"{path}.lora.b"] = jnp.zeros((lora_spec[path], arr.shape[1]), dtype=arr.dtype)
            freezed[f"{path}.lora.alpha"] = alpha
            freezed[path] = arr
    trainable = flax.traverse_util.unflatten_dict(trainable, sep=".")

    return trainable, freezed
