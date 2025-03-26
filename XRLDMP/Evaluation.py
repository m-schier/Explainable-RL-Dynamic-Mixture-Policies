import gymnasium as gym
import numpy as np
from typing import Callable, Optional


def evaluate(env: gym.Env, act_fn, episodes: int = 100, record_video: bool = False, etr_threshold: float = -.5,
             compound_metrics: Optional[Callable[[dict], dict]] = None):
    returns = []
    lengths = []
    successes = []
    etrs = []
    frames = []
    metrics = {}

    for episode in range(episodes):
        current_return = 0.
        current_length = 0

        # Reseed with episode number before each episode to have as deterministic environment sequence as possible
        obs, _ = env.reset(seed=episode)
        early_termination = False
        success = False

        while True:
            act = act_fn(obs)

            if episode == 0 and record_video:
                frames.append(env.render())

            obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            current_return += reward
            current_length += 1

            if done:
                # Log all final metrics
                for k, v in info.items():
                    if k.startswith('FinalMetric.') or k.startswith('Metric.'):
                        metric_key = k[k.find('.') + 1:]
                        metrics[metric_key] = metrics.get(metric_key, []) + [v]
                if not truncated:
                    if reward < etr_threshold:
                        early_termination = True
                    else:
                        success = True
                break

        etrs.append(early_termination)
        successes.append(success)
        returns.append(current_return)
        lengths.append(current_length)

    log_dict = {
        "avg_return": np.mean(returns),
        "avg_episode_length": np.mean(lengths),
        "avg_etr": np.mean(etrs),
        "avg_sr": np.mean(successes),
        **{f"avg_{k}": np.mean(v) for k, v in metrics.items()},
    }

    if compound_metrics is not None:
        log_dict = {**log_dict, **compound_metrics(log_dict)}

    if record_video:
        import wandb

        try:
            dt = env.unwrapped.dt
        except AttributeError:
            dt = env.get_wrapper_attr('dt')

        frames = np.stack(frames)
        frames = np.transpose(frames, (0, 3, 1, 2))
        log_dict['video'] = wandb.Video(frames, fps=int(1 / dt), format='mp4')

    return log_dict
