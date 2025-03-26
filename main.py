import sys

import chex
import jax
# Fixes broken softmax backward
jax.config.update('jax_softmax_custom_jvp', True)

import hydra
import omegaconf
import jax.numpy as jnp
import numpy as np
import gymnasium as gym
from CarEnv import CarEnv
from XRLDMP.Architectures import FlatFeatureExtractor, ComposedFeatureExtractor, DeepSetFeatureExtractor, FourierFeatureNetwork
from XRLDMP.Buffers import DictBuffer
from XRLDMP.Evaluation import evaluate
from XRLDMP.RolloutMetrics import RolloutMetrics
from XRLDMP.Util import optimize_set_batch
from functools import partial


@jax.jit
def batch_obs(obs):
    return jax.tree_util.tree_map(lambda x: x[None, ...], obs)


def make_deepset_fe(cfg: omegaconf.DictConfig):
    from functools import partial

    kwargs = dict(
        item_encoder_units=(cfg.feature_extractor.width,) * 3,
        set_encoder_units=(cfg.feature_extractor.width,) * 2,
    )

    if cfg.feature_extractor.set_ffn is False:
        cones_set = partial(DeepSetFeatureExtractor, **kwargs)
    elif isinstance(cfg.feature_extractor.set_ffn, float):
        cones_set = partial(DeepSetFeatureExtractor, **kwargs,
                            preencoder=partial(FourierFeatureNetwork, b_scale=cfg.feature_extractor.set_ffn))
    else:
        raise ValueError(f"{cfg.feature_extractor.set_ffn = }")

    if cfg.problem.type == "racing":
        object_key = "cones_set"
    elif cfg.problem.type == "racing_tracking":
        object_key = "trajectory_set"
    else:
        raise ValueError(f"{cfg.problem.type = }")

    spec = {
        object_key: cones_set,
        'state': partial(FlatFeatureExtractor, n_hidden=128),
    }

    return ComposedFeatureExtractor(spec=spec)


def positional_encoding(x, freqs=4, include_linear=True):
    factors = jnp.pi * jnp.power(2., jnp.arange(freqs))[:, None]

    result = jnp.concatenate([
        jnp.cos(factors * x[..., None, :]),
        jnp.sin(factors * x[..., None, :])
    ], axis=-2)

    if include_linear:
        result = jnp.concatenate([result, x[..., None, :]], axis=-2)

    result = result.reshape(x.shape[:-1] + (-1,))
    return result


def make_tracking_fe(cfg: omegaconf.DictConfig):
    assert cfg.problem.type == 'racing_tracking' and cfg.feature_extractor.type == 'tracking_flat'

    preproc = (lambda x: x) if cfg.feature_extractor.pe_freqs == 0 else \
        partial(positional_encoding, freqs=cfg.feature_extractor.pe_freqs, include_linear=cfg.feature_extractor.pe_orig)

    def _trajectory_stub():
        def forward(x):
            x = x[..., 2:]  # Cut present flag and positional
            chex.assert_axis_dimension(x, -1, 2)
            result = x.reshape(x.shape[:-2] + (-1,))  # Flatten last two axes (n_objects x n_features)
            return preproc(result)

        return forward

    return ComposedFeatureExtractor(spec={
        'trajectory_set': _trajectory_stub,
        'state': lambda: lambda x: preproc(x),
    })


def make_fe(cfg: omegaconf.DictConfig):
    if cfg.feature_extractor.type == "deepset":
        return make_deepset_fe(cfg)
    elif cfg.feature_extractor.type == "tracking_flat":
        return make_tracking_fe(cfg)
    else:
        raise ValueError(f"{cfg.feature_extractor.type = }")


def make_agent(cfg: omegaconf.DictConfig, env):
    from functools import partial

    m_fe = partial(make_fe, cfg)

    if cfg.agent.type == 'xdsac':
        from XRLDMP.ExplainableDSAC import make_xdsac
        from XRLDMP.Architectures import make_actor_obs_reshaper
        from XRLDMP.ConfigParsing import parse_xdsac

        feature_reshaper_actor = make_actor_obs_reshaper(cfg)
        kwargs = parse_xdsac(cfg, batch_obs)
        return make_xdsac(feature_reshaper_actor, m_fe, env.action_space, **kwargs), True
    elif cfg.agent.type == 'xsac':
        from XRLDMP.ExplainableSAC import make_xsac
        from XRLDMP.Architectures import make_actor_obs_reshaper
        from XRLDMP.ConfigParsing import parse_xsac

        feature_reshaper_actor = make_actor_obs_reshaper(cfg)
        kwargs = parse_xsac(cfg)

        return make_xsac(feature_reshaper_actor, m_fe, env.action_space, **kwargs), False
    elif cfg.agent.type == 'dsac':
        from XRLDMP.DSAC import make_dsac
        from XRLDMP.ConfigParsing import parse_dsac
        kwargs = parse_dsac(cfg, batch_obs)
        return make_dsac(m_fe, env.action_space, **kwargs), True
    elif cfg.agent.type == 'sac':
        from XRLDMP.SAC import make_sac
        from XRLDMP.ConfigParsing import parse_sac
        kwargs = parse_sac(cfg)
        return make_sac(m_fe, env.action_space, **kwargs), False
    else:
        raise ValueError(f"{cfg.agent.type = }")


def make_env_producer(cfg: omegaconf.DictConfig, render_mode=None, callbacks=None):
    from XRLDMP.Wrappers import ObsCheckWrapper

    is_discrete = cfg.agent.type not in ('sac', 'xsac')

    if cfg.problem.type == "racing":
        from XRLDMP.Wrappers import CarEnvDiscreteWrapper, CarEnvContinuousWrapper
        from CarEnv.Configs import get_standard_env_config
        if is_discrete:
            return lambda: ObsCheckWrapper(CarEnvDiscreteWrapper(CarEnv(get_standard_env_config("racing"), render_mode=render_mode,
                                           render_kwargs={'hints': {'callbacks': callbacks or []}})))
        else:
            if cfg.problem.reduced_actions:
                return lambda: ObsCheckWrapper(CarEnvContinuousWrapper(CarEnv(get_standard_env_config("racing"), render_mode=render_mode,
                                               render_kwargs={'hints': {'callbacks': callbacks or []}})))
            else:
                return lambda: ObsCheckWrapper(CarEnv(get_standard_env_config("racing"), render_mode=render_mode,
                                               render_kwargs={'hints': {'callbacks': callbacks or []}}))
    elif cfg.problem.type == "racing_tracking":
        from XRLDMP.Wrappers import CarEnvDiscreteWrapper, CarEnvContinuousWrapper, CarEnvTrajectoryToSetWrapper
        from CarEnv.Configs import get_standard_env_config

        config = get_standard_env_config("racing_tracking")
        config['sensors']['trajectory']['lookahead_points'] = cfg.problem.lookahead_points
        config['sensors']['trajectory']['step'] = cfg.problem.step

        # To have the dots corresponding to the action explanation
        draw_sensors = bool(callbacks)

        if is_discrete:
            return lambda: ObsCheckWrapper(CarEnvDiscreteWrapper(CarEnvTrajectoryToSetWrapper(CarEnv(config, render_mode=render_mode,
                                           render_kwargs={'hints': {'draw_sensors': draw_sensors, 'callbacks': callbacks or []}}))))
        else:
            if cfg.problem.reduced_actions:
                return lambda: ObsCheckWrapper(CarEnvContinuousWrapper(CarEnvTrajectoryToSetWrapper(CarEnv(config, render_mode=render_mode,
                                               render_kwargs={'hints': {'draw_sensors': draw_sensors, 'callbacks': callbacks or []}}))))
            else:
                return lambda: ObsCheckWrapper(CarEnvTrajectoryToSetWrapper(CarEnv(config, render_mode=render_mode,
                                               render_kwargs={'hints': {'draw_sensors': draw_sensors, 'callbacks': callbacks or []}})))
    else:
        raise ValueError(f"{cfg.problem.type = }")


def stateful_stochastic_act(agent, rng):
    split2 = jax.jit(jax.random.split)

    def stub(state, x):
        nonlocal rng
        rng, act_rng = split2(rng)
        return agent.act_stochastic(state, x, act_rng)
    return stub


def make_eval_policy(agent, rng):
    from XRLDMP.ExplainableSAC import XSACImpl

    if isinstance(agent, XSACImpl):
        # argmax is non-trivial, sample policy during evaluation
        return stateful_stochastic_act(agent, rng)
    else:
        return agent.act_deterministic


def make_callbacks(cfg: omegaconf.DictConfig):
    from XRLDMP.Callbacks import SingleDiscreteExplainableRenderCallback, SingleContinuousExplainableRenderCallback

    if cfg.agent.type.startswith('xdsac') and cfg.problem.type == "racing":
        render_cb = SingleDiscreteExplainableRenderCallback()
        callbacks = [render_cb]
    elif cfg.agent.type == 'xsac' and cfg.problem.type == "racing":
        render_cb = SingleContinuousExplainableRenderCallback()
        callbacks = [render_cb]
    elif cfg.agent.type == 'xsac' and cfg.problem.type == "racing_tracking":
        from XRLDMP.Callbacks import TrackingContinuousExplainableRenderCallback
        render_cb = TrackingContinuousExplainableRenderCallback()
        callbacks = [render_cb]
    elif cfg.agent.type.startswith('xdsac') and cfg.problem.type == "racing_tracking":
        from XRLDMP.Callbacks import TrackingExplainableRenderCallback
        render_cb = TrackingExplainableRenderCallback()
        callbacks = [render_cb]
    else:
        render_cb = None
        callbacks = []
        render_deterministic = None

    return render_cb, callbacks


@hydra.main(config_path="conf", config_name="base.yaml")
def main(cfg: omegaconf.DictConfig):
    from tqdm import tqdm
    import os
    import wandb
    import sys
    import pickle
    from XRLDMP.RolloutWorker import ParallelRolloutWorker, Transition

    print(f"{cfg = }", file=sys.stderr)

    os.makedirs('tmp/', exist_ok=True)
    wandb.init(dir="tmp/", project='XRLDMP',
               config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    eval_frequency = cfg.get('eval_frequency', 100_000)

    render_cb, callbacks = make_callbacks(cfg)

    env_prod = make_env_producer(cfg, render_mode=None, callbacks=callbacks)
    rollout_worker = ParallelRolloutWorker(env_prod, 4)

    compound_metrics = None

    try:
        eval_env = make_env_producer(cfg, render_mode='rgb_array', callbacks=callbacks)()

        print(f"{rollout_worker.observation_space = }", file=sys.stderr)
        print(f"{rollout_worker.observation_space.shape = }", file=sys.stderr)
        print(f"{rollout_worker.action_space = }", file=sys.stderr)

        seed = np.random.randint(2 ** 32 - 1)
        numpy_rng = np.random.default_rng(seed)
        rng = jax.random.PRNGKey(seed)
        split3 = jax.jit(partial(jax.random.split, num=3))
        rng, init_rng, buffer_rng = split3(rng)

        dsac, is_discrete = make_agent(cfg, rollout_worker)
        example_obs = rollout_worker.example_obs()
        dsac_state = dsac.init(init_rng, example_obs)

        act_fn = make_eval_policy(dsac, rng)

        rollout_metrics = RolloutMetrics()

        buffer = DictBuffer(rollout_worker.observation_space, rollout_worker.action_space, cfg.agent.buffer_size)

        # Warm start
        for _ in tqdm(range(cfg.agent.batch_size)):
            transition: Transition = rollout_worker.collect_transition()

            rollout_worker.act_on_last_transition(rollout_worker.action_space.sample())
            rollout_metrics.update(transition.reward, transition.terminated, transition.truncated,
                               episode_length=transition.episode_length, episode_return=transition.episodic_return)

            if not transition.truncated:
                buffer.put(transition.obs, transition.next_obs, transition.act, transition.reward, transition.terminated)

        # Real training
        for step in tqdm(range(cfg.agent.batch_size, cfg.steps)):
            transition: Transition = rollout_worker.collect_transition()
            rollout_metrics.update(transition.reward, transition.terminated, transition.truncated,
                               episode_length=transition.episode_length, episode_return=transition.episodic_return)

            if not transition.truncated:
                buffer.put(transition.obs, transition.next_obs, transition.act, transition.reward, transition.terminated)

            rng, act_rng, train_rng = jax.random.split(rng, 3)

            if is_discrete and numpy_rng.uniform() < .05:
                act = rollout_worker.action_space.sample()
            else:
                act = dsac.act_stochastic(dsac_state, transition.next_obs, act_rng)
                act = np.asarray(act)

            rollout_worker.act_on_last_transition(act)

            batch = buffer.sample(cfg.agent.batch_size, numpy_rng)
            batch = optimize_set_batch(batch, freeze=False)

            if is_discrete:
                dsac_state_new, train_metrics = dsac.train_step(dsac_state, batch)
            else:
                dsac_state_new, train_metrics = dsac.train_step(dsac_state, batch, train_rng)

            dsac_state = dsac_state_new

            if step % 2000 == 0 and step > 0:
                wandb.log({
                    **rollout_metrics.report(),
                    **{f"train/{k}": v for k, v in train_metrics.items() if v is not None},
                }, step=step)

            # Eval
            if step % eval_frequency == 0 and step > 0:
                if render_cb is not None:
                    render_cb.policy = lambda x: dsac.act_deterministic(dsac_state, x, return_full=True)[1]

                wandb.log({f"train_eval/{k}": v for k, v in
                        evaluate(eval_env, partial(act_fn, dsac_state), record_video=True, compound_metrics=compound_metrics).items()}, step=step)
    finally:
        rollout_worker.close()
        del rollout_worker

    # Save model
    with open(os.path.join(wandb.run.dir, "model.pckl"), 'wb') as fp:
        pickle.dump({
            "config": cfg,
            "state": dsac_state,
        }, fp)

    # Final eval
    final_log = evaluate(eval_env, partial(act_fn, dsac_state), record_video=True, compound_metrics=compound_metrics)
    wandb.log({
        **{f"eval/{k}": v for k, v in final_log.items()},
        **{f"train_eval/{k}": v for k, v in final_log.items()},
    }, step=cfg.steps)


if __name__ == '__main__':
    main()
