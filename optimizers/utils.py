from typing import NamedTuple

import chex


class Decomposition(NamedTuple):
    matu: chex.Array  # shape=(d, tau), dtype=jnp.floating.
    sigma: chex.Array  # shape=(tau,), dtype=jnp.floating.
