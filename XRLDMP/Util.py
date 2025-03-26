import flax
import numba
import numpy as np


@numba.jit(nopython=True)
def _find_size(obs, next_obs, steps=1):
    batch_size, max_obs, n_features = obs.shape
    assert obs.shape == next_obs.shape

    # max(np.max(np.sum(v[..., 0] > 0, axis=-1)), np.max(np.sum(v_next[..., 0] > 0, axis=-1)))

    for i in range(max_obs):
        if not np.any(obs[:, i, 0] > 0) and not np.any(next_obs[:, i, 0] > 0):
            break
    else:
        i = max_obs

    return min(max_obs, int(np.ceil(i / steps) * steps))


def optimize_set_batch(batch, freeze=True, steps=8):
    # Optimize the size of an encoded set by cutting of rows of empty items for the entire batch.
    # Since any change in shape requires a recompilation of the XLA kernel be a lot less aggressive than we could be.
    # Thus, we currently jointly optimize obs and next_obs and use a certain step.

    obs, next_obs, *rest = batch

    new_size = {}

    for k, v in obs.items():
        if not k.endswith('_set'):
            continue

        v_next = next_obs[k]

        # Assumes present items contiguous from 0
        new_size[k] = _find_size(v, v_next, steps=steps)

    obs = {k: v[:, :new_size[k]] if k in new_size else v for k, v in obs.items()}
    next_obs = {k: v[:, :new_size[k]] if k in new_size else v for k, v in next_obs.items()}

    if freeze:
        obs = flax.core.FrozenDict(obs)
        next_obs = flax.core.FrozenDict(next_obs)

    return obs, next_obs, *rest
