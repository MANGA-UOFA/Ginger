from optax import adamw, sgd

from optimizers.ginger import ginger  # noqa: F401
from optimizers.qng import qng

__optimizers__ = {
    "adamw": adamw,
    "ginger": ginger,
    "sgd": sgd,
    "qng": qng,
}


def get_optimizer_cls(name):
    try:
        return __optimizers__[name]
    except KeyError as e:
        raise ValueError(f"Unknown optimizer {name}") from e


__ALL__ = __optimizers__.keys()
