import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import chex
import gymnasium as gym

from typing import Callable, Tuple, Optional
from .SAC import default_batch_obs, get_loss, polyak_update


def dsac_check_action_space(space: gym.Space):
    if not isinstance(space, gym.spaces.Discrete):
        raise TypeError(type(space))
    return space.n


class CategoricalPolicy(nn.Module):
    n_act: int
    fe_producer: Callable[[], nn.Module]
    net_arch: Tuple[int] = (256, 256)

    @nn.compact
    def __call__(self, inp):
        x = self.fe_producer()(inp)

        for f in self.net_arch:
            x = nn.Dense(features=f)(x)
            x = nn.relu(x)

        # Duelling Head ("Dueling Network Architectures for Deep Reinforcement Learning")
        logits = nn.Dense(features=self.n_act)(x)
        logits = logits - jnp.mean(logits, axis=-1, keepdims=True)
        return logits


class DiscreteCritic(nn.Module):
    n_act: int
    fe_producer: Callable[[], nn.Module]
    n_critics: int = 2
    net_arch: Tuple[int] = (256, 256)
    duelling: bool = True

    @nn.compact
    def __call__(self, inp):
        features = self.fe_producer()(inp)

        result = []

        for _ in range(self.n_critics):
            x = features
            for f in self.net_arch:
                x = nn.Dense(features=f)(x)
                x = nn.relu(x)

            # Duelling Head ("Dueling Network Architectures for Deep Reinforcement Learning")
            if self.duelling:
                adv = nn.Dense(features=self.n_act)(x)
                val = nn.Dense(features=1)(x)
                result.append(adv - jnp.mean(adv, axis=-1, keepdims=True) + val)
            else:
                result.append(nn.Dense(features=self.n_act)(x))

        return jnp.stack(result, axis=0)


@chex.dataclass(frozen=True)
class DsacState:
    variables_actor: None
    variables_critic: None
    variables_critic_target: None
    log_ent_coef: None
    opt_state_actor: None
    opt_state_critic: None
    opt_state_ent_coef: None


@chex.dataclass(frozen=True)
class DsacTrainMetrics:
    loss_actor: None
    loss_critic: None
    loss_ent_coef: Optional[float]
    grad_norm_actor: None
    grad_norm_critic: None
    entropy_mean: None
    entropy_std: None
    ent_coef: None
    q_val_min: float
    q_val_max: float


@chex.dataclass(frozen=True)
class DsacImpl:
    init: None
    act_deterministic: None
    act_stochastic: None
    train_step: None


def make_dsac(fe_producer: Callable, action_space: gym.Space, net_arch: Tuple[int] = (256, 256), collate_fn=None,
              n_critics=2, alpha=.015, target_entropy=None, lr: float = 1e-4, gamma: float = .99, loss='huber', tau: float = 5e-3,
              clip_norm: Optional[float] = None, duelling: bool = True):
    """
    Implement a Soft Actor-Critic on Discrete action spaces, roughly following "Soft Actor-Critic for Discrete Action
    Settings", Christodoulou, with some common changes from modern SAC implementations like omitting the value network.
    :param fe_producer: Feature extractor producer
    :param action_space: Action space of the environment, must be discrete
    :param net_arch: Net architectures in units per layer
    :param collate_fn: Function to create a batch of one observation from a single observation for act* functions
    :param n_critics: Number of critics, 2 in standard SAC and paper
    :param alpha: Weighting of the entropy term in the objective
    :param lr: Learning rate for both actor and critic
    :param gamma: Discount factor
    :param loss: Loss for critic. Paper uses L2, huber sometimes considered more stable. Actor uses gradient ascent.
    :param tau: Step size for critic target update, default from SB3 SAC, SAC-Discrete paper uses hard updates
    :param clip_norm: Global norm to clip gradients by, no clipping if None
    :return: Implementation of DSAC
    """
    n_act = dsac_check_action_space(action_space)
    critic = DiscreteCritic(n_act=n_act, fe_producer=fe_producer, net_arch=net_arch, n_critics=n_critics, duelling=duelling)
    actor = CategoricalPolicy(n_act=n_act, fe_producer=fe_producer, net_arch=net_arch)
    collate_fn = collate_fn or default_batch_obs
    loss_fn = get_loss(loss)

    if target_entropy is None:
        target_entropy = 0.9 * jnp.log(n_act)

    if clip_norm is None:
        optimizer_actor = optax.adam(learning_rate=lr)
        optimizer_critic = optax.adam(learning_rate=lr)
    else:
        optimizer_actor = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=lr))
        optimizer_critic = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=lr))

    optimizer_ent_coef = optax.adam(learning_rate=lr)

    def init(rng: jax.Array, obs) -> DsacState:
        variables_actor = actor.init(rng, collate_fn(obs))
        variables_critic = critic.init(rng, collate_fn(obs))
        variables_critic_target = jax.tree_map(lambda x: x.copy(), variables_critic)
        opt_state_actor = optimizer_actor.init(variables_actor['params'])
        opt_state_critic = optimizer_critic.init(variables_critic['params'])

        if alpha == "auto":
            from .SAC import sac_init_log_entropy_coefficient
            log_ent_coef = sac_init_log_entropy_coefficient()
            opt_state_ent_coef = optimizer_ent_coef.init(log_ent_coef)
        else:
            log_ent_coef = jnp.log(alpha)
            opt_state_ent_coef = None

        return DsacState(
            variables_actor=variables_actor,
            variables_critic=variables_critic,
            variables_critic_target=variables_critic_target,
            opt_state_actor=opt_state_actor,
            opt_state_critic=opt_state_critic,
            opt_state_ent_coef=opt_state_ent_coef,
            log_ent_coef=log_ent_coef,
        )

    @jax.jit
    def act_stochastic(state: DsacState, obs, rng):
        logits = actor.apply(state.variables_actor, obs)
        chex.assert_equal(logits.shape[-1], n_act)
        return jax.random.categorical(rng, logits, axis=-1)

    @jax.jit
    def act_deterministic(state: DsacState, obs):
        logits = actor.apply(state.variables_actor, obs)
        chex.assert_equal(logits.shape[-1], n_act)
        return jnp.argmax(logits, axis=-1)

    @jax.jit
    def train_step(state: DsacState, batch) -> Tuple[DsacState, DsacTrainMetrics]:
        obs, next_obs, action, reward, done = batch
        n, = reward.shape
        chex.assert_equal_shape([action, reward, done])

        # Find action distribution for next state using online policy
        next_logits = actor.apply(state.variables_actor, next_obs)
        next_probs = jax.nn.softmax(next_logits, axis=-1)
        next_entropy = -jnp.sum(jnp.log(jnp.clip(next_probs, 1e-10, 1)) * next_probs, axis=-1)
        chex.assert_shape(next_entropy, (n,))

        # Calculate expectation for Q-vals using offline policy (use worst estimator)
        next_qvals = critic.apply(state.variables_critic_target, next_obs)
        next_qvals = jnp.min(next_qvals, axis=0)
        chex.assert_shape(next_qvals, (n, n_act))
        next_expected_return = jnp.sum(next_qvals * next_probs, axis=-1)
        chex.assert_shape(next_expected_return, (n,))

        target = reward + (1 - done) * gamma * (next_expected_return + jnp.exp(state.log_ent_coef) * next_entropy)
        chex.assert_shape(target, (n,))

        c_vars, c_params_init = flax.core.pop(state.variables_critic, 'params')

        def critic_objective(p):
            current_q_vals = critic.apply({**c_vars, 'params': p}, obs)
            current_q_vals = current_q_vals[..., jnp.arange(n), action]
            chex.assert_shape(current_q_vals, (n_critics, n))

            return loss_fn(current_q_vals, jnp.broadcast_to(target, current_q_vals.shape))

        loss_critic, grads = jax.value_and_grad(critic_objective)(c_params_init)
        grad_norm_critic = optax.global_norm(grads)
        updates, new_opt_state_critic = optimizer_critic.update(grads, state.opt_state_critic, c_params_init)
        new_variables_critic = {**c_vars, 'params': optax.apply_updates(c_params_init, updates)}

        a_vars, a_params_init = flax.core.pop(state.variables_actor, 'params')

        def actor_objective(p):
            logits = actor.apply({**a_vars, 'params': p}, obs)

            mean_q_vals = jnp.mean(critic.apply(new_variables_critic, obs), axis=0)
            chex.assert_shape(mean_q_vals, (n, n_act))

            probs = jax.nn.softmax(logits, axis=-1)
            entropy = -jnp.sum(jnp.log(jnp.clip(probs, 1e-10, 1)) * probs, axis=-1)
            chex.assert_shape(entropy, (n,))

            expected_return = jnp.sum(mean_q_vals * probs, axis=-1)
            chex.assert_shape(expected_return, (n,))

            return -jnp.mean(expected_return + jnp.exp(state.log_ent_coef) * entropy), entropy

        (loss_actor, aux_entropy), grads = jax.value_and_grad(actor_objective, has_aux=True)(a_params_init)
        grad_norm_actor = optax.global_norm(grads)
        updates, new_opt_state_actor = optimizer_actor.update(grads, state.opt_state_actor, a_params_init)
        new_variables_actor = {**a_vars, 'params': optax.apply_updates(a_params_init, updates)}

        # Update entropy coefficient
        def ent_coef_objective(p):
            return -jnp.mean(p * (-aux_entropy + target_entropy))

        if alpha == "auto":
            loss_ent_coef, grads = jax.value_and_grad(ent_coef_objective)(state.log_ent_coef)
            updates, new_optimizer_state_ent_coef = optimizer_ent_coef.update(grads, state.opt_state_ent_coef, state.log_ent_coef)
            new_log_ent_coef = optax.apply_updates(state.log_ent_coef, updates)
        else:
            loss_ent_coef = None
            new_optimizer_state_ent_coef = state.optimizer_state_ent_coef
            new_log_ent_coef = state.log_ent_coef

        # Move target
        new_variables_critic_target = polyak_update(new_variables_critic, state.variables_critic_target, tau)

        return state.replace(
            variables_actor=new_variables_actor,
            variables_critic=new_variables_critic,
            variables_critic_target=new_variables_critic_target,
            opt_state_actor=new_opt_state_actor,
            opt_state_critic=new_opt_state_critic,
            opt_state_ent_coef=new_optimizer_state_ent_coef,
            log_ent_coef=new_log_ent_coef,
        ), DsacTrainMetrics(
            loss_actor=loss_actor,
            loss_critic=loss_critic,
            loss_ent_coef=loss_ent_coef,
            grad_norm_actor=grad_norm_actor,
            grad_norm_critic=grad_norm_critic,
            entropy_mean=jnp.mean(aux_entropy),
            entropy_std=jnp.sqrt(jnp.mean(jnp.square(aux_entropy - jnp.mean(aux_entropy)))),
            ent_coef=jnp.exp(state.log_ent_coef),
            q_val_min=jnp.min(target),
            q_val_max=jnp.max(target),
        )

    return DsacImpl(
        init=init,
        act_deterministic=act_deterministic,
        act_stochastic=act_stochastic,
        train_step=train_step,
    )
