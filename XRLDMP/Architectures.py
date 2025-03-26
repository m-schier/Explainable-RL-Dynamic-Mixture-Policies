import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import chex
from functools import partial
from typing import Callable, Optional, Tuple


class FourierFeatureNetwork(nn.Module):
    features: int
    b_scale: float = 1.

    @staticmethod
    def bias_init(key, shape, dtype) -> jax.Array:
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, -1., 1.)

    @nn.compact
    def __call__(self, x):
        kernel_init = flax.linen.initializers.normal(stddev=self.b_scale / x.shape[-1])
        return jnp.sin(nn.Dense(features=self.features, kernel_init=kernel_init, bias_init=self.bias_init)(x) * jnp.pi)


class ComposedFeatureExtractor(nn.Module):
    spec: None

    @nn.compact
    def __call__(self, inp):
        # FrozenDict required for now but may not be needed in the future, see
        # https://github.com/google/flax/discussions/3191
        # TODO: If a leaf is marked None, we should not rely on it existing just to ignore it
        x = jax.tree_map(lambda s, a: s()(a) if s is not None else None, self.spec, flax.core.FrozenDict(inp))
        return jax.tree_util.tree_reduce(lambda a, b: jnp.concatenate([a, b], axis=-1) if b is not None else a, x)


class DeepSetFeatureExtractor(nn.Module):
    item_encoder_units: Tuple[int] = (64, 32, 128)
    set_encoder_units: Tuple[int] = (128, 128)
    preencoder: Optional[Callable] = None
    kernel_init: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        kernel_init = self.kernel_init or nn.initializers.lecun_normal()

        present = x[..., :1]

        # Item transforms
        x = x[..., 1:]

        for i, f in enumerate(self.item_encoder_units):
            if i == 0 and self.preencoder is not None:
                x = self.preencoder(features=f)(x)
            else:
                x = nn.Dense(features=f, kernel_init=kernel_init)(x)
                x = nn.relu(x)

        # Aggregate items
        # Be hopeful that one day XLA will compile all the useless items away
        x = jax.lax.select(jnp.broadcast_to(present > 0, x.shape), x, jnp.zeros_like(x))
        x = jnp.sum(x, axis=-2)

        # Set transforms
        for f in self.set_encoder_units:
            x = nn.Dense(features=f, kernel_init=kernel_init)(x)
            x = nn.relu(x)

        return x


class FlatFeatureExtractor(nn.Module):
    n_hidden: int = 128

    @nn.compact
    def __call__(self, x):
        return nn.relu(nn.Dense(features=self.n_hidden)(x))


def actor_obs_reshaper(x, object_key="cones_set", add_neutral=False):
    x, state = flax.core.pop(x, 'state')
    x, objects = flax.core.pop(x, object_key)

    chex.assert_equal(len(x), 0)
    chex.assert_equal(len(state.shape) + 1, len(objects.shape))

    present = objects[..., :1] > 0

    if add_neutral:
        *rest, _, feature_dim = present.shape
        present = jnp.concatenate([jnp.ones((*rest, 1, feature_dim), present.dtype), present], axis=-2)
        *rest, _, feature_dim = objects.shape
        objects = jnp.concatenate([jnp.zeros((*rest, 1, feature_dim), objects.dtype), objects], axis=-2)

    # Make state concatenable
    state = state[..., None, :]
    state = jnp.broadcast_to(state, objects.shape[:-1] + (state.shape[-1],))

    entries = jnp.concatenate([objects, state], axis=-1)

    return entries, present


def make_actor_obs_reshaper(cfg):
    if cfg.problem.type == "racing":
        return partial(actor_obs_reshaper, object_key="cones_set")
    elif cfg.problem.type == "racing_tracking":
        return partial(actor_obs_reshaper, object_key="trajectory_set")
    else:
        raise ValueError(f"{cfg.problem.type = }")
