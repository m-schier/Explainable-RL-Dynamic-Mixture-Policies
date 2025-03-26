import sys
from multiprocessing import Process
from dataclasses import dataclass
from typing import Any, Dict
import gymnasium
from pettingzoo import ParallelEnv


try:
    from faster_fifo import Queue
    _WORKER_READ_TIMEOUT = -1.  # None not supported for faster_fifo, uses negative number instead
except ImportError:
    print("RolloutWorker: Could not import faster_fifo, performance may be worse", file=sys.stderr)
    from multiprocessing import Queue
    _WORKER_READ_TIMEOUT = None  # Negative number not supported by multiprocessing, uses None


@dataclass(frozen=True)
class Transition:
    obs: Any
    next_obs: Any
    act: Any
    reward: float
    episodic_return: float
    episode_length: int
    terminated: bool
    truncated: bool


class AbstractRolloutWorker:
    def collect_transition(self) -> Transition:
        """
        Collect a new transition from the worker
        :return: Transition
        """
        raise NotImplementedError

    def example_obs(self) -> Any:
        """
        Return an exemplary observation that may be used to initialize architecture, etc.
        :return: Observation
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the worker and free resources
        """
        pass

    def act_on_last_transition(self, act: Any) -> None:
        """
        Select the next action to take for the rollout with the latest transition
        :param act: Action
        """
        raise NotImplementedError


def _run_worker(env_producer, idx: int, seed: int, data_queue: Queue, command_queue: Queue):
    try:
        env = env_producer()
        episodic_return = 0.
        episode_length = 0

        obs, _ = env.reset(seed=seed)

        # Sample first action randomly to keep interface simple
        act = env.action_space.sample()

        while True:
            next_obs, reward, terminated, truncated, _ = env.step(act)
            episodic_return += reward
            episode_length += 1

            if truncated or terminated:
                next_obs, _ = env.reset()

            data_queue.put((idx, Transition(obs=obs, next_obs=next_obs, act=act, reward=reward,
                                            episodic_return=episodic_return, episode_length=episode_length,
                                            terminated=terminated, truncated=truncated)))

            if truncated or terminated:
                episodic_return = 0.
                episode_length = 0

            obs = next_obs
            act = command_queue.get(block=True, timeout=_WORKER_READ_TIMEOUT)

            if act is None:
                return

    except Exception as ex:
        from traceback import print_exc
        print_exc()
        data_queue.put(ex)


class ParallelRolloutWorker(AbstractRolloutWorker):
    def __init__(self, producer, n_envs):
        import numpy as np

        self._dummy_env = producer()
        self.data_queue = Queue()
        self.command_queues = []
        self.processes = []
        self._active_idx = None

        seed = np.random.randint(0, 2 ** 31)

        for idx in range(n_envs):
            cq = Queue()
            p = Process(target=_run_worker, daemon=True, args=(producer, idx, seed + idx, self.data_queue, cq))
            self.command_queues.append(cq)
            self.processes.append(p)
            p.start()

    def example_obs(self):
        obs, _ = self._dummy_env.reset()
        return obs

    def close(self):
        for q in self.command_queues:
            q.put(None)

        for p in self.processes:
            p.join()

    def collect_transition(self):
        assert self._active_idx is None

        data = self.data_queue.get(block=True, timeout=30.)

        if isinstance(data, Exception):
            raise data
        else:
            self._active_idx, transition = data

        return transition

    def act_on_last_transition(self, act):
        assert self._active_idx is not None

        self.command_queues[self._active_idx].put(act)
        self._active_idx = None

    def __getattr__(self, item):
        # Prevent a common error case
        if item in ("reset", "step"):
            raise AttributeError("Trying to access a blocked method from the environment dummy")

        return getattr(self._dummy_env, item)


class SerialRolloutWorker(AbstractRolloutWorker):
    """
    A wrapper for an Env on the same thread such that the interface remains coherent with AsyncEnv.
    """
    def __init__(self, env_producer):
        self._env = env_producer()
        self._obs, _ = self._env.reset()
        self._dummy_obs = self._obs
        self._last_act = self._env.action_space.sample()
        self._episodic_return = 0.
        self._episode_length = 0

    def example_obs(self):
        return self._dummy_obs

    def collect_transition(self):
        assert self._last_act is not None

        next_obs, reward, terminated, truncated, _ = self._env.step(self._last_act)
        self._episodic_return += reward
        self._episode_length += 1

        if terminated or truncated:
            next_obs, _ = self._env.reset()

        result = Transition(obs=self._obs, next_obs=next_obs, act=self._last_act, reward=reward,
                            episodic_return=self._episodic_return, episode_length=self._episode_length,
                            terminated=terminated, truncated=truncated)

        if terminated or truncated:
            self._episodic_return = 0.
            self._episode_length = 0

        self._last_act = None
        self._obs = next_obs

        return result

    def act_on_last_transition(self, act):
        assert self._last_act is None
        self._last_act = act

    def __getattr__(self, item):
        return getattr(self._env, item)

    def close(self):
        pass


@dataclass(frozen=True)
class MarlTransition:
    obs: Any
    next_obs: Any
    act: Any
    reward: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class MarlRolloutStep:
    active: Dict[str, Any]  # Dictionary of all currently active agents and their observations
    transitions: Dict[str, MarlTransition]  # Dictionary of all transitions from the last step
    was_reset: bool  # Whether the env was completely reset for this step


class AbstractMarlRolloutWorker:
    def collect_transition(self) -> MarlRolloutStep:
        """
        Collect a new transition from the worker
        :return: Transition
        """
        raise NotImplementedError

    def single_agent_action_space(self) -> gymnasium.Space:
        raise NotImplementedError

    def example_obs(self) -> Any:
        """
        Return an exemplary observation that may be used to initialize architecture, etc.
        :return: Observation
        """
        raise NotImplementedError

    def example_single_agent_obs(self) -> Any:
        """
        Return an exemplary observation of a single agent
        :return: Observation of single agent
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close the worker and free resources
        """
        pass

    def act_on_last_transition(self, act: Dict[str, Any]) -> None:
        """
        Select the next action to take for the rollout with the latest transition
        :param act: Action
        """
        raise NotImplementedError


class SerialMarlRolloutWorker(AbstractMarlRolloutWorker):
    """
    A wrapper for an Env on the same thread such that the interface remains coherent with AsyncEnv.
    """
    def __init__(self, env_producer, reset_criterion):
        self._env: ParallelEnv = env_producer()
        self._reset_criterion = reset_criterion
        self._obs, _ = self._env.reset()
        self._dummy_obs = self._obs
        self._last_act: Dict[str, Any] = {k: self._env.action_space(k).sample() for k in self._obs}

    def single_agent_action_space(self) -> gymnasium.Space:
        return self._env.action_space(self._env.possible_agents[0])

    def example_obs(self):
        return self._dummy_obs

    def example_single_agent_obs(self):
        return self._env.observation_space(self._env.possible_agents[0]).sample()

    def collect_transition(self):
        assert self._last_act is not None

        next_obs, reward, terminated, truncated, _ = self._env.step(self._last_act)

        transition_dict = {}

        for k, o in self._obs.items():
            transition_dict[k] = MarlTransition(o, next_obs[k], self._last_act[k], reward[k], terminated[k], truncated[k])

        self._last_act = None  # Not strictly required, but useful to keep track of state

        do_reset = self._reset_criterion(self._env)

        if do_reset:
            active_obs, _ = self._env.reset()
        else:
            active_obs = {k: v for k, v in next_obs.items() if not (terminated[k] or truncated[k])}

        self._obs = dict(active_obs)  # Shallow copy

        return MarlRolloutStep(active_obs, transition_dict, do_reset)

    def act_on_last_transition(self, act):
        assert self._last_act is None
        self._last_act = act

    def __getattr__(self, item):
        return getattr(self._env, item)

    def close(self):
        pass


def _run_worker_marl(env_producer, reset_criterion, idx: int, seed: int, data_queue: Queue, command_queue: Queue):
    try:
        env = env_producer()

        obs, _ = env.reset(seed=seed)

        # Sample first action randomly to keep interface simple
        act = {k: env.action_space(k).sample() for k in obs}

        while True:
            next_obs, reward, terminated, truncated, _ = env.step(act)

            transition_dict = {}

            for k, o in obs.items():
                transition_dict[k] = MarlTransition(o, next_obs[k], act[k], reward[k], terminated[k], truncated[k])

            do_reset = reset_criterion(env)

            if do_reset:
                next_obs, _ = env.reset()
            else:
                next_obs = {k: v for k, v in next_obs.items() if not (terminated[k] or truncated[k])}

            active_obs = dict(next_obs)  # Shallow copy

            data_queue.put((idx, MarlRolloutStep(active_obs, transition_dict, do_reset)))

            obs = next_obs
            act = command_queue.get(block=True, timeout=_WORKER_READ_TIMEOUT)

            if act is None:
                return

    except Exception as ex:
        from traceback import print_exc
        print_exc()
        data_queue.put(ex)


class ParallelMarlRolloutWorker(AbstractMarlRolloutWorker):
    def __init__(self, env_producer, reset_criterion, n_envs):
        import numpy as np

        self._dummy_env: ParallelEnv = env_producer()
        self.data_queue = Queue()
        self.command_queues = []
        self.processes = []
        self._active_idx = None

        seed = np.random.randint(0, 2 ** 31)

        for idx in range(n_envs):
            cq = Queue()
            p = Process(target=_run_worker_marl, daemon=True, args=(env_producer, reset_criterion, idx, seed + idx, self.data_queue, cq))
            self.command_queues.append(cq)
            self.processes.append(p)
            p.start()

    def single_agent_action_space(self) -> gymnasium.Space:
        return self._dummy_env.action_space(self._dummy_env.possible_agents[0])

    def example_obs(self):
        obs, _ = self._dummy_env.reset()
        return obs

    def example_single_agent_obs(self):
        return self._dummy_env.observation_space(self._dummy_env.possible_agents[0]).sample()

    def close(self):
        for q in self.command_queues:
            q.put(None)

        for p in self.processes:
            p.join()

    def collect_transition(self) -> MarlRolloutStep:
        assert self._active_idx is None

        data = self.data_queue.get(block=True, timeout=30.)

        if isinstance(data, Exception):
            raise data
        else:
            self._active_idx, step_data = data

        return step_data

    def act_on_last_transition(self, act):
        assert self._active_idx is not None

        self.command_queues[self._active_idx].put(act)
        self._active_idx = None

    def __getattr__(self, item):
        # Prevent a common error case
        if item in ("reset", "step"):
            raise AttributeError("Trying to access a blocked method from the environment dummy")

        return getattr(self._dummy_env, item)
