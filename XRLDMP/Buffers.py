import flax.core
import gymnasium as gym
import numpy as np


def _realize_zeros(*args, **kwargs):
    # np.zeros() internally uses calloc(), which may be implemented to map all virtual pages to a single physical
    # page containing only zeros with copy-on-write semantics. This is usually not desirable for our use-case, because:
    #  1. We immediately want to fail if the machine would not have enough physical memory for all buffers
    #  2. The unexpected physical memory growth during training might create the impression of a memory-leak
    # Thus, immediately force CoW on the entire array.

    x = np.zeros(*args, **kwargs)
    x[:] = 0
    return x


class DictBuffer:
    def __init__(self, obs_space: gym.spaces.Space, act_space: gym.spaces.Space, n: int):
        if not isinstance(obs_space, gym.spaces.Dict):
            raise TypeError(type(obs_space))

        for k, v in obs_space.items():
            if not isinstance(v, gym.spaces.Box):
                raise TypeError(f"{k}: {type(v)}")

        if isinstance(act_space, gym.spaces.Discrete):
            act_shape = tuple()
            act_dtype = 'int32'
        elif isinstance(act_space, gym.spaces.Box):
            act_shape = act_space.shape
            act_dtype = 'float32'
        else:
            raise TypeError(type(act_space))

        self.obs = {k: _realize_zeros((n,) + v.shape, dtype=v.dtype) for k, v in obs_space.items()}
        self.next_obs = {k: _realize_zeros((n,) + v.shape, dtype=v.dtype) for k, v in obs_space.items()}
        self.action = _realize_zeros((n,) + act_shape, dtype=act_dtype)
        self.reward = _realize_zeros((n,), dtype='float32')
        self.done = _realize_zeros((n,), dtype='float32')
        self.count = 0
        self.next_idx = 0
        self.capacity = n

    def __repr__(self):
        def np_repr(arr):
            assert isinstance(arr, np.ndarray)
            return f"np.ndarray(shape = {arr.shape}, dtype = {arr.dtype})"

        def dict_repr(d):
            return "{" + ", ".join([f"{k}: {np_repr(v) if isinstance(v, np.ndarray) else v}" for k, v in d.items()]) + "}"

        return f"DictBuffer(obs = {dict_repr(self.obs)}, next_obs = {dict_repr(self.next_obs)}, " \
               f"action = {np_repr(self.action)}, reward = {np_repr(self.reward)}, done = {np_repr(self.done)}, " \
               f"count = {self.count}, next_idx = {self.next_idx})"

    def put(self, obs, next_obs, action, reward, done):
        for k in self.obs.keys():
            self.obs[k][self.next_idx] = obs[k]
            self.next_obs[k][self.next_idx] = next_obs[k]

        self.action[self.next_idx] = action
        self.reward[self.next_idx] = reward
        self.done[self.next_idx] = done

        self.next_idx = (self.next_idx + 1) % self.capacity

        if self.count < self.capacity:
            self.count += 1

    def sample(self, batch_size, rng: np.random.Generator, freeze=False):
        idxs = rng.integers(0, self.count, batch_size)

        obs = {k: v[idxs] for k, v in self.obs.items()}
        next_obs = {k: v[idxs] for k, v in self.next_obs.items()}

        if freeze:
            obs = flax.core.FrozenDict(obs)
            next_obs = flax.core.FrozenDict(next_obs)

        return obs, next_obs, self.action[idxs], self.reward[idxs], self.done[idxs]
