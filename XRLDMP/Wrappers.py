import gymnasium as gym
import numpy as np
import sys
from gymnasium.core import ObsType, WrapperObsType


class CarEnvDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        from itertools import product
        super(CarEnvDiscreteWrapper, self).__init__(env)

        if env.action_space.shape == (3,):
            self.action_mapping = list(product(np.linspace(-1, 1, 3), np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)))
        elif env.action_space.shape == (2,):
            self.action_mapping = list(product(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3)))
        else:
            raise ValueError(f"{env.action_space = }")
        print(f"{self.action_mapping = }", file=sys.stderr)
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))

    def action(self, action):
        action = np.asarray(action)

        assert np.all((action >= 0) & (action <= len(self.action_mapping)))

        return np.array(self.action_mapping[action])


class CarEnvContinuousWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(CarEnvContinuousWrapper, self).__init__(env)
        self.action_space = gym.spaces.Box(-1, 1, (2,))

    def action(self, action):
        steering, longitudinal = action

        assert -1 <= longitudinal <= 1

        if longitudinal >= 0.:
            throttle = longitudinal * 2 - 1
            brake = -1
        else:
            throttle = -1
            brake = -longitudinal * 2 - 1

        return np.array([steering, throttle, brake])


class CarEnvTrajectoryToSetWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CarEnvTrajectoryToSetWrapper, self).__init__(env)
        traj_features, = env.observation_space['trajectory'].shape
        self.observation_space = gym.spaces.Dict({
            **{k: v for k, v in env.observation_space.items() if k != 'trajectory'},
            'trajectory_set': gym.spaces.Box(-1, 1, (traj_features // 2, 4))
        })

    @staticmethod
    def set_transform(observation):
        traj_obs = observation['trajectory'].reshape(-1, 2)
        present = np.ones(len(traj_obs))
        indices = np.arange(len(traj_obs)) / (len(traj_obs - 1)) * 2 - 1  # Cover range [-1, 1]
        new_traj_obs = np.concatenate([present[..., None], indices[..., None], traj_obs], axis=-1)
        return {
            **{k: v for k, v in observation.items() if k != 'trajectory'},
            'trajectory_set': new_traj_obs
        }

    def observation(self, observation: ObsType) -> WrapperObsType:
        return CarEnvTrajectoryToSetWrapper.set_transform(observation)


class MultiAgentCarEnvDiscreteActionMapper:
    def __init__(self):
        from itertools import product

        self.action_mapping = list(product(np.linspace(-1, 1, 3), np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)))
        print(f"{self.action_mapping = }", file=sys.stderr)
        self.single_agent_action_space = gym.spaces.Discrete(len(self.action_mapping))

    def map_single(self, action):
        return self.action_mapping[action]

    def action(self, action):
        return {k: np.array(self.action_mapping[v]) for k, v in action.items()}


class MultiAgentCarEnvPseudoMapper:
    def __init__(self):
        self.single_agent_action_space = gym.spaces.Box(-1, 1, (3,))

    def map_single(self, action):
        return action

    def action(self, action):
        return action


class ObsCheckWrapper(gym.ObservationWrapper):
    def observation(self, observation: ObsType):
        for k, v in observation.items():
            if not np.all(np.isfinite(v)):
                raise ValueError(f"Non-finite values in {k}")

            ma, mi = np.max(v), np.min(v)

            if ma > 1. or mi < -1.:
                raise ValueError(f"Bad input range {mi} - {ma} in {k}")

        return observation
