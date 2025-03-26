import numpy as np
from typing import Dict, Optional
import time


class RolloutMetrics:
    def __init__(self, episode_stats_window_size=100, step_stats_window_size=10000):
        self.episode_return_window = np.zeros(episode_stats_window_size)
        self.episode_length_window = np.zeros(episode_stats_window_size)
        self.step_reward_window = np.zeros(step_stats_window_size)
        self.step_dt_window = np.zeros(step_stats_window_size)
        self.episode_count: int = 0
        self.episode_head: int = 0
        self.step_count: int = 0
        self.step_head: int = 0
        self.current_episode_length: int = 0
        self.current_episode_return: float = 0.
        self.last_time = None

    def update(self, reward: float, terminated: bool, truncated: bool,
               episode_length: Optional[int] = None, episode_return: Optional[float] = None):
        # Also count length on truncation, such that we can achieve maximum steps of the env
        self.current_episode_length += 1

        curr_time = time.time()

        if not truncated:
            # Update step metrics
            self.current_episode_return += reward
            self.step_reward_window[self.step_head] = reward
            self.step_dt_window[self.step_head] = curr_time - self.last_time if self.last_time is not None else 0.
            self.step_head = (self.step_head + 1) % self.step_reward_window.shape[0]
            self.step_count = min(self.step_count + 1, self.step_reward_window.shape[0])

        self.last_time = curr_time

        if terminated or truncated:
            effective_episode_length = episode_length or self.current_episode_length
            effective_episode_return = episode_return or self.current_episode_return

            self.episode_return_window[self.episode_head] = effective_episode_return
            self.episode_length_window[self.episode_head] = effective_episode_length
            self.episode_head = (self.episode_head + 1) % self.episode_length_window.shape[0]
            self.episode_count = min(self.episode_count + 1, self.episode_length_window.shape[0])
            self.current_episode_return = 0.
            self.current_episode_length = 0

    def report(self) -> Dict[str, float]:
        return {
            'rollout/avg_step_reward': np.mean(self.step_reward_window[:max(1, self.step_count)]).item(),
            'rollout/avg_episode_return': np.mean(self.episode_return_window[:max(1, self.episode_count)]).item(),
            'rollout/avg_episode_length': np.mean(self.episode_length_window[:max(1, self.episode_count)]).item(),
            'rollout/sps': 1. / np.mean(self.step_dt_window[:max(1, self.step_count)]).item()
        }


class MarlRolloutMetrics:
    def __init__(self):
        self._tracked_agents = {}
        self._episode_lengths = []
        self._episode_returns = []
        self._step_times = []
        self._last_time = None

    def reset_tracked(self):
        for k in list(self._tracked_agents.keys()):
            self._add_episode(k)
        self._tracked_agents = {}

    def update(self, reward, terminated, truncated):
        curr_time = time.time()
        if self._last_time is not None:
            self._step_times.append(curr_time - self._last_time)
        while len(self._step_times) > 5000:
            self._step_times.pop(0)
        self._last_time = curr_time

        for k in reward.keys():
            if k not in self._tracked_agents:
                self._tracked_agents[k] = [reward[k]]
            else:
                self._tracked_agents[k].append(reward[k])

            if truncated[k] or terminated[k]:
                self._add_episode(k)

    def update_from_transitions(self, transitions):
        curr_time = time.time()
        if self._last_time is not None:
            self._step_times.append(curr_time - self._last_time)
        while len(self._step_times) > 5000:
            self._step_times.pop(0)
        self._last_time = curr_time

        for k, tran in transitions.items():
            if k not in self._tracked_agents:
                self._tracked_agents[k] = [tran.reward]
            else:
                self._tracked_agents[k].append(tran.reward)

            if tran.truncated or tran.terminated:
                self._add_episode(k)

    def _add_episode(self, k):
        ag_rewards = self._tracked_agents.pop(k)
        self._episode_lengths.append(len(ag_rewards))
        self._episode_returns.append(np.sum(ag_rewards) if ag_rewards else 0.)
        self._episode_lengths = self._episode_lengths[-250:]
        self._episode_returns = self._episode_returns[-250:]

    def report(self) -> Dict[str, float]:
        return {
            'rollout/avg_episode_return': np.mean(self._episode_returns).item() if self._episode_returns else 0.,
            'rollout/avg_episode_length': np.mean(self._episode_lengths).item() if self._episode_lengths else 0,
            'rollout/sps': 1. / np.mean(self._step_times) if self._step_times else 0.,
        }
