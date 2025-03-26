import sys

import sys
import os

import jax
from pettingzoo.utils.env import AgentID, ActionType, ObsType

# Fixes broken softmax backward
jax.config.update('jax_softmax_custom_jvp', True)

import hydra
import omegaconf
import numpy as np
from functools import partial
from typing import Optional, Callable
import wandb
import pettingzoo
import gymnasium


def marl_fe(cfg):
    from XRLDMP.Architectures import FlatFeatureExtractor, ComposedFeatureExtractor, DeepSetFeatureExtractor

    assert cfg.feature_extractor.type == "deepset"

    width = cfg.feature_extractor.width

    assert not cfg.feature_extractor.set_ffn

    return ComposedFeatureExtractor(spec={
        'vehicles_set': partial(DeepSetFeatureExtractor, item_encoder_units=(width, width, width), set_encoder_units=(width, width)),
        'state': partial(FlatFeatureExtractor, n_hidden=width),
    })


def eval_marl(env, act_fn, episodes: int = 20, record_video: bool = False, stop_criterion: Optional[Callable] = None):
    from tqdm import tqdm
    metrics = {}
    frames = []

    if stop_criterion is None:
        stop_criterion = lambda my_env: my_env.time > 60.

    for i in tqdm(range(episodes), desc="Evaluating"):
        obs, info = env.reset(seed=i)
        active_agents = list(obs.keys())

        total_return = 0.

        while not stop_criterion(env):
            if i == 0 and record_video:
                frames.append(env.render())

            actions = act_fn({k: obs[k] for k in active_agents})
            obs, reward, terminated, truncated, info = env.step(actions)
            active_agents = [k for k in obs.keys() if not (terminated[k] or truncated[k])]

            total_return += sum(reward.values())

        for k, v in info.get(None, {}).items():
            if k.startswith('FinalMetric.') or k.startswith('Metric.'):
                metric_key = k[k.find('.') + 1:]
                metrics[metric_key] = metrics.get(metric_key, []) + [v]
        metrics['TotalEpisodicReturn'] = metrics.get('TotalEpisodicReturn', []) + [total_return]

    log_dict = {
        **{f"avg_{k}": np.mean(v) for k, v in metrics.items()},
    }

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


def make_marl_agent(cfg: omegaconf.DictConfig, action_space, obs_space=None):
    from main import batch_obs
    from XRLDMP.Architectures import actor_obs_reshaper

    fe_prod = partial(marl_fe, cfg)

    if cfg.agent.type == 'xdsac':
        from XRLDMP.ConfigParsing import parse_xdsac
        from XRLDMP.ExplainableDSAC import make_xdsac

        feature_reshaper_actor = partial(actor_obs_reshaper, object_key="vehicles_set", add_neutral=True)
        kwargs = parse_xdsac(cfg, batch_obs)
        return make_xdsac(feature_reshaper_actor, fe_prod, action_space, **kwargs)
    elif cfg.agent.type == 'dsac':
        from XRLDMP.ConfigParsing import parse_dsac
        from XRLDMP.DSAC import make_dsac
        kwargs = parse_dsac(cfg, batch_obs)
        return make_dsac(fe_prod, action_space, **kwargs)
    elif cfg.agent.type == 'sac':
        from XRLDMP.ConfigParsing import parse_sac
        from XRLDMP.SAC import make_sac
        kwargs = parse_sac(cfg)
        return make_sac(fe_prod, action_space, **kwargs)
    elif cfg.agent.type == 'xsac':
        from XRLDMP.ConfigParsing import parse_xsac
        from XRLDMP.ExplainableSAC import make_xsac

        feature_reshaper_actor = partial(actor_obs_reshaper, object_key="vehicles_set", add_neutral=True)
        kwargs = parse_xsac(cfg)

        return make_xsac(feature_reshaper_actor, fe_prod, action_space, **kwargs)
    else:
        raise ValueError(f"{cfg.agent.type = }")


def stack_obs(obs):
    result = {}

    for k in obs[0].keys():
        result[k] = np.stack([o[k] for o in obs], axis=0)

    return result


def act_on_mapping(act_fn, obs):
    actor_list = list(obs.keys())

    if not actor_list:
        return {}

    encoded_obs = {}

    for k in obs[actor_list[0]].keys():
        encoded_obs[k] = np.stack([obs[actor][k] for actor in actor_list], axis=0)

    action_list = np.asarray(act_fn(encoded_obs))

    return {actor: action for actor, action in zip(actor_list, action_list)}


def get_mapper(cfg):
    from XRLDMP.Wrappers import MultiAgentCarEnvDiscreteActionMapper, MultiAgentCarEnvPseudoMapper

    if cfg.agent.type in ('xdsac', 'dsac'):
        mapper = MultiAgentCarEnvDiscreteActionMapper()
        is_discrete = True
    elif cfg.agent.type in ('xsac', 'sac'):
        mapper = MultiAgentCarEnvPseudoMapper()
        is_discrete = False
    else:
        raise ValueError(f"{cfg.agent.type = }")

    return mapper, is_discrete


class MarlEnvCurriculumResetWrapper(pettingzoo.utils.wrappers.BaseParallelWrapper):
    def __init__(self, env, curriculum, attribute='max_vehicles'):
        super(MarlEnvCurriculumResetWrapper, self).__init__(env)
        self.curriculum = curriculum
        self.curriculum_attribute = attribute

    def reset(self, *args, **kwargs):
        choice_probs = 1. / np.asarray(self.curriculum)
        choice_probs /= np.sum(choice_probs)
        setattr(self.env.problem, self.curriculum_attribute, self.unwrapped.np_random.choice(self.curriculum, p=choice_probs))
        return super().reset(*args, **kwargs)


class MarlEnvDiscreteActionWrapper(pettingzoo.utils.wrappers.BaseParallelWrapper):
    def __init__(self, env):
        from itertools import product
        super(MarlEnvDiscreteActionWrapper, self).__init__(env)
        self.__action_mapping = list(product(np.linspace(-1, 1, 3), np.linspace(-1, 1, 2), np.linspace(-1, 1, 2)))

    def action_space(self, agent: AgentID) -> gymnasium.spaces.Space:
        return gymnasium.spaces.Discrete(len(self.__action_mapping))

    def step(self, actions: dict[AgentID, ActionType]) -> tuple[
        dict[AgentID, ObsType],
        dict[AgentID, float],
        dict[AgentID, bool],
        dict[AgentID, bool],
        dict[AgentID, dict],
    ]:
        actions = {k: np.array(self.__action_mapping[v]) for k, v in actions.items()}
        return super().step(actions)


def make_env_producer(cfg: omegaconf.DictConfig, render_mode=None, render_kwargs=None):
    from CarEnv.Configs import MARL_INTERSECTION
    from copy import deepcopy
    from CarEnv.MultiAgentEnv import MultiAgentCarEnv

    _, is_discrete = get_mapper(cfg)

    if cfg.problem.type == "marl_intersection":
        env_cfg = deepcopy(MARL_INTERSECTION)
        env_cfg['problem']['k_tracking'] = 0.015

        if cfg.problem.soft_collisions:
            env_cfg['problem']['soft_collision_pad'] = (.5, .25)

        curriculum_vehicles = [1, 2, 4, 8]
        # Limit max sensor objects to expected max vehicles to save buffer-memory
        env_cfg['sensors']['vehicles_set']['max_objects'] = max(curriculum_vehicles)

        env_producer = lambda: MarlEnvCurriculumResetWrapper(
            MultiAgentCarEnv(env_cfg, render_mode=render_mode, render_kwargs=render_kwargs), curriculum_vehicles)
        eval_env_producer = lambda **kwargs: MarlEnvCurriculumResetWrapper(
            MultiAgentCarEnv(env_cfg, **kwargs), [max(curriculum_vehicles)])

        stop_criterion = lambda my_env: my_env.time >= 60.
    else:
        raise ValueError(f"{cfg.problem.type = }")

    if is_discrete:
        base_env_producer = env_producer
        base_eval_env_producer = eval_env_producer
        env_producer = lambda: MarlEnvDiscreteActionWrapper(base_env_producer())
        eval_env_producer = lambda **kwargs: MarlEnvDiscreteActionWrapper(base_eval_env_producer(**kwargs))

    return env_producer, env_cfg, eval_env_producer, stop_criterion


def make_render_callback(cfg: omegaconf.DictConfig):
    from XRLDMP.Callbacks import MARLExplainableRenderCallback

    if cfg.agent.type == 'xdsac':
        render_cb = MARLExplainableRenderCallback()
        callbacks = [render_cb]
    elif cfg.agent.type == 'xsac':
        render_cb = MARLExplainableRenderCallback(discrete=False)
        callbacks = [render_cb]
    else:
        render_cb = None
        callbacks = []

    return render_cb, callbacks


@hydra.main(config_path="conf", config_name="marl_intersection_xdsac.yaml")
def main(cfg: omegaconf.DictConfig):
    import pickle
    from tqdm import tqdm
    from main import make_eval_policy

    from XRLDMP.RolloutMetrics import MarlRolloutMetrics
    from XRLDMP.RolloutWorker import ParallelMarlRolloutWorker
    from XRLDMP.Buffers import DictBuffer
    from XRLDMP.Util import optimize_set_batch

    seed = np.random.randint(2 ** 32 - 1)
    numpy_rng = np.random.default_rng(seed)

    env_producer, env_cfg, eval_env_producer, stop_criterion = make_env_producer(cfg)
    _, is_discrete = get_mapper(cfg)

    render_cb, callbacks = make_render_callback(cfg)

    optimize_batch = True

    rollout_worker = ParallelMarlRolloutWorker(env_producer, stop_criterion, 2)

    try:
        rng = jax.random.PRNGKey(seed)
        split3 = jax.jit(partial(jax.random.split, num=3))
        split2 = jax.jit(jax.random.split)
        rng, init_rng, buffer_rng = split3(rng)

        reset_steps = cfg.get('reset_steps', None)
        print(f"{reset_steps = }", file=sys.stderr)

        eval_env = eval_env_producer(render_mode='rgb_array', render_kwargs={'width': 1280, 'height': 1280, 'hints': {'callbacks': callbacks}})

        single_action_space = eval_env.action_space(eval_env.possible_agents[0])
        agent = make_marl_agent(cfg, single_action_space, eval_env.observation_space(eval_env.possible_agents[0]))
        init_obs = rollout_worker.example_single_agent_obs()
        agent_state = agent.init(init_rng, init_obs)

        act_fn = make_eval_policy(agent, rng)

        rollout_metrics = MarlRolloutMetrics()
        buffer = DictBuffer(rollout_worker.observation_space(rollout_worker.possible_agents[0]), single_action_space,
                            cfg.agent.buffer_size)
        print(f"{buffer = }", file=sys.stderr)

        # Logging
        os.makedirs('tmp/', exist_ok=True)
        wandb.init(dir="tmp/", project='XRLDMP',
                   config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

        warm_start = 2000
        eval_frequency = cfg.get('eval_frequency', 100_000)

        for step in tqdm(range(0, cfg.steps), "Training MARL"):
            # Resetting: The Primacy Bias in Deep Reinforcement Learning
            if reset_steps is not None and step % reset_steps == 0:
                # Scrub the agent
                rng, init_rng = jax.random.split(rng)
                agent_state = agent.init(init_rng, init_obs)

            rollout_step = rollout_worker.collect_transition()

            # Action selection
            if step >= warm_start:
                actions = {}

                if rollout_step.active:
                    rng, act_rng = split2(rng)
                    actions = act_on_mapping(lambda x: agent.act_stochastic(agent_state, x, act_rng), rollout_step.active)

                if is_discrete:
                    actions = {k: single_action_space.sample() if numpy_rng.uniform() < .05 else v for k, v in actions.items()}
            else:
                actions = {k: single_action_space.sample() for k in rollout_step.active}

            # Collecting experience
            rollout_worker.act_on_last_transition(actions)

            rollout_metrics.update_from_transitions(rollout_step.transitions)

            if rollout_step.was_reset:
                rollout_metrics.reset_tracked()

            experiences_added = 0
            for tran in rollout_step.transitions.values():
                if not tran.truncated:
                    buffer.put(tran.obs, tran.next_obs, tran.act, tran.reward, tran.terminated)
                    experiences_added += 1

            # Training
            train_metrics = {}
            if step >= warm_start:
                for _ in range(experiences_added):
                    batch = buffer.sample(cfg.agent.batch_size, numpy_rng)
                    if optimize_batch:
                        batch = optimize_set_batch(batch, freeze=False)
                    train_rng = None

                    if is_discrete:
                        new_agent_state, train_metrics = agent.train_step(agent_state, batch)
                    else:
                        rng, train_rng = split2(rng)
                        new_agent_state, train_metrics = agent.train_step(agent_state, batch, train_rng)

                    agent_state = new_agent_state

            if step > 0 and step % 2000 == 0:
                wandb.log({
                    "train/buffer_count": buffer.count,
                    **rollout_metrics.report(),
                    **{f"train/{k}": v for k, v in train_metrics.items() if v is not None}
                }, step=step)

            if step > 0 and step % eval_frequency == 0:
                if render_cb is not None:
                    render_cb.policy = lambda x: agent.act_deterministic(agent_state, x, return_full=True)[1]
                wandb.log({f"train_eval/{k}": v for k, v in
                           eval_marl(eval_env, lambda o: act_on_mapping(partial(act_fn, agent_state), o), record_video=True, stop_criterion=stop_criterion).items()},
                          step=step)
    finally:
        rollout_worker.close()
        del rollout_worker

    # Save model
    with open(os.path.join(wandb.run.dir, "model.pckl"), 'wb') as fp:
        pickle.dump({
            "config": cfg,
            "state": agent_state,
        }, fp)

    # Final eval
    if render_cb is not None:
        render_cb.policy = lambda x: agent.act_deterministic(agent_state, x, return_full=True)[1]
    final_log = eval_marl(eval_env, lambda o: act_on_mapping(partial(act_fn, agent_state), o), record_video=True, stop_criterion=stop_criterion)
    wandb.log({
        **{f"eval/{k}": v for k, v in final_log.items()},
        **{f"train_eval/{k}": v for k, v in final_log.items()},
    }, step=cfg.steps)


if __name__ == '__main__':
    main()
