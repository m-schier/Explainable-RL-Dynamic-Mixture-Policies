import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import chex
import gymnasium as gym

from functools import partial
from typing import Callable, Tuple, Optional
from .SAC import default_batch_obs, get_loss, polyak_update, sac_init_log_entropy_coefficient
from .DSAC import dsac_check_action_space, DiscreteCritic, DsacState


def sample_categorical_probs(key, probs):
    cum = jnp.cumsum(probs, axis=-1)

    # Normalizes probs such that total is 1
    cum = cum / cum[..., -1:]

    rands = jax.random.uniform(key, shape=cum.shape[:-1] + (1,))

    return jnp.sum(rands >= cum, axis=-1)


class CategoricalMixturePolicy(nn.Module):
    n_act: int
    observation_shaper: Callable
    uniform_mixture: bool = False
    net_arch: Tuple[int] = (256, 256, 256, 256)

    @nn.compact
    def __call__(self, inp):
        x, present = self.observation_shaper(inp)

        chex.assert_equal(present.shape, x.shape[:-1] + (1,))

        for f in self.net_arch:
            x = nn.Dense(features=f)(x)
            x = nn.relu(x)

        adv = nn.Dense(features=self.n_act)(x)
        adv = adv - jnp.mean(adv, axis=-1, keepdims=True)

        if self.uniform_mixture:
            mixture_weight_logits = jnp.zeros_like(x[..., :1])
        else:
            mixture_weight_logits = nn.Dense(features=1)(x)
            mixture_weight_logits = mixture_weight_logits - jnp.mean(mixture_weight_logits, axis=-2, keepdims=True)

        chex.assert_equal_shape([mixture_weight_logits, present])
        mixture_weights = jax.nn.softmax(mixture_weight_logits, axis=-2, where=present, initial=-jnp.inf)

        # If no object is given (which is not allowed!), fall back to safe behavior
        valid_mixture = jnp.sum(mixture_weights, axis=-2, keepdims=True) > 0.
        mixture_weights = jax.lax.select(jnp.broadcast_to(valid_mixture, mixture_weights.shape), mixture_weights,
                                         jnp.ones_like(mixture_weights) / mixture_weights.shape[-2])

        # We are returning logits for softmax, so zeros like any constant are uniform and thus have no influence
        object_logits = jax.lax.select(jnp.broadcast_to(present, adv.shape), adv, jnp.zeros_like(adv))
        object_probs = jax.nn.softmax(object_logits, axis=-1)

        probs = jnp.sum(mixture_weights * object_probs, axis=-2)
        chex.assert_shape(probs, present.shape[:-2] + (self.n_act,))

        return {
            'object_logits': object_logits,
            'object_probs': object_probs,
            'probs': probs,
            'mixture_logits': mixture_weight_logits,
            'weights': mixture_weights,
        }


@chex.dataclass(frozen=True)
class XDSACTrainMetrics:
    loss_actor: None
    loss_critic: None
    loss_ent_coef: Optional[float]
    grad_norm_actor: None
    grad_norm_critic: None
    mixture_entropy: None
    mean_mixture_logits: None
    entropy_mean: None
    entropy_std: None
    ent_coef: None
    kappa: None
    q_val_max: float
    q_val_min: float


@chex.dataclass(frozen=True)
class XDSACImpl:
    init: None
    act_deterministic: None
    act_stochastic: None
    train_step: None


@chex.dataclass(frozen=True)
class XDsacState:
    variables_actor: None
    variables_critic: None
    variables_critic_target: None
    opt_state_actor: None
    opt_state_critic: None
    opt_state_ent_coef: None
    log_ent_coef: None


def make_xdsac(feature_reshaper_actor: Callable, fe_producer_critic: Callable, action_space: gym.Space,
               actor_net_arch: Tuple[int] = (256, 256), critic_net_arch: Tuple[int] = (256, 256), collate_fn=None,
               n_critics=2, alpha=.015, lr: float = 1e-4, q_lr: Optional[float] = None, gamma: float = .99,
               loss='huber', tau: float = 5e-3, uniform_mixture: bool = False, kappa: float = .01,
               clip_norm: Optional[float] = None, target_entropy=None):
    """
    A spin on an explainable DSAC on dynamic observation spaces using myopic actors.
    :param feature_reshaper_actor: The (usually not learnable) function that reshapes the observation for the
    myopic per-object actors.
    :param fe_producer_critic: Feature extractor producer
    :param action_space: Action space of the environment, must be discrete
    :param actor_net_arch: Net architectures in units per layer for the actor
    :param critic_net_arch: Net architectures in units per layer for the critics
    :param collate_fn: Function to create a batch of one observation from a single observation for act* functions
    :param n_critics: Number of critics, 2 in standard SAC and paper
    :param alpha: Weighting of the entropy term in the objective
    :param lr: Learning rate for actor and critic, if q_lr unspecified
    :param q_lr: Learning rate for critic if specified, otherwise critic uses same lr as actor
    :param gamma: Discount factor
    :param loss: Loss for critic. Paper uses L2, huber sometimes considered more stable. Actor uses gradient ascent.
    :param tau: Step size for critic target update, default from SB3 SAC, SAC-Discrete paper uses hard updates
    :return: Implementation of DSAC
    """
    n_act = dsac_check_action_space(action_space)
    critic = DiscreteCritic(n_act=n_act, fe_producer=fe_producer_critic, net_arch=critic_net_arch, n_critics=n_critics)
    actor = CategoricalMixturePolicy(n_act=n_act, observation_shaper=feature_reshaper_actor, net_arch=actor_net_arch,
                                     uniform_mixture=uniform_mixture)
    collate_fn = collate_fn or default_batch_obs
    loss_fn = get_loss(loss)

    q_lr = q_lr or lr
    target_entropy = target_entropy or jnp.log(n_act) * .5

    if clip_norm is None:
        optimizer_actor = optax.adam(learning_rate=lr)
        optimizer_critic = optax.adam(learning_rate=q_lr)
    else:
        optimizer_actor = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=lr))
        optimizer_critic = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=q_lr))

    optimizer_ent_coef = optax.adam(learning_rate=q_lr)

    def init(rng: jax.Array, obs) -> XDsacState:
        variables_actor = actor.init(rng, collate_fn(obs))
        variables_critic = critic.init(rng, collate_fn(obs))
        variables_critic_target = jax.tree_map(lambda x: x.copy(), variables_critic)
        opt_state_actor = optimizer_actor.init(variables_actor['params'])
        opt_state_critic = optimizer_critic.init(variables_critic['params'])

        if alpha == "auto":
            log_ent_coef = sac_init_log_entropy_coefficient()
            opt_state_ent_coef = optimizer_ent_coef.init(log_ent_coef)
        else:
            log_ent_coef = jnp.log(alpha)
            opt_state_ent_coef = None

        return XDsacState(
            variables_actor=variables_actor,
            variables_critic=variables_critic,
            variables_critic_target=variables_critic_target,
            opt_state_actor=opt_state_actor,
            opt_state_critic=opt_state_critic,
            opt_state_ent_coef=opt_state_ent_coef,
            log_ent_coef=log_ent_coef,
        )

    @partial(jax.jit, static_argnums=(3,))
    def act_stochastic(state: DsacState, obs, rng, return_full=False):
        act_result = actor.apply(state.variables_actor, obs)
        probs = act_result['probs']
        chex.assert_equal(probs.shape[-1], n_act)
        choice = sample_categorical_probs(rng, probs)
        chex.assert_equal(probs.shape[:-1], choice.shape)

        if return_full:
            return choice, act_result
        else:
            return choice

    @partial(jax.jit, static_argnums=(2,))
    def act_deterministic(state: DsacState, obs, return_full=False):
        act_result = actor.apply(state.variables_actor, obs)
        probs = act_result['probs']
        chex.assert_equal(probs.shape[-1], n_act)
        choice = jnp.argmax(probs, axis=-1)
        chex.assert_equal(probs.shape[:-1], jnp.shape(choice))

        if return_full:
            return choice, act_result
        else:
            return choice

    @jax.jit
    def train_step(state: XDsacState, batch) -> Tuple[XDsacState, XDSACTrainMetrics]:
        obs, next_obs, action, reward, done = batch
        n = len(reward)

        # Find action distribution for next state using online policy
        next_probs = actor.apply(state.variables_actor, next_obs)['probs']
        chex.assert_shape(next_probs, (n, n_act))
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
            mean_q_vals = jnp.mean(critic.apply(new_variables_critic, obs), axis=0)
            chex.assert_shape(mean_q_vals, (n, n_act))
            act_result = actor.apply({**a_vars, 'params': p}, obs)
            mix_probs = jnp.squeeze(act_result['weights'], axis=-1)
            mixture_entropy = -jnp.sum(jnp.log(jnp.clip(mix_probs, 1e-10, 1)) * mix_probs, axis=-1)
            chex.assert_shape(mixture_entropy, (n,))
            probs = act_result['probs']
            chex.assert_shape(probs, (n, n_act))
            entropy = -jnp.sum(jnp.log(jnp.clip(probs, 1e-10, 1)) * probs, axis=-1)
            chex.assert_shape(entropy, (n,))

            expected_return = jnp.sum(mean_q_vals * probs, axis=-1)
            chex.assert_shape(expected_return, (n,))

            return -jnp.mean(expected_return + jnp.exp(state.log_ent_coef) * entropy + kappa * mixture_entropy), (entropy, mixture_entropy, jnp.mean(act_result['mixture_logits']))

        (loss_actor, (aux_entropy, aux_mixture_entropy, aux_mean_mixture_logits)), grads = jax.value_and_grad(actor_objective, has_aux=True)(a_params_init)
        grad_norm_actor = optax.global_norm(grads)
        updates, new_opt_state_actor = optimizer_actor.update(grads, state.opt_state_actor, a_params_init)
        new_variables_actor = {**a_vars, 'params': optax.apply_updates(a_params_init, updates)}

        # Optionally update entropy
        def ent_coef_objective(p):
            return -jnp.mean(p * (-aux_entropy + target_entropy))

        if alpha == "auto":
            loss_ent_coef, grads = jax.value_and_grad(ent_coef_objective)(state.log_ent_coef)
            updates, new_opt_state_ent_coef = optimizer_ent_coef.update(grads, state.opt_state_ent_coef, state.log_ent_coef)
            new_log_ent_coef = optax.apply_updates(state.log_ent_coef, updates)
        else:
            loss_ent_coef = None
            new_opt_state_ent_coef = state.opt_state_ent_coef
            new_log_ent_coef = state.log_ent_coef

        # Move target
        new_variables_critic_target = polyak_update(new_variables_critic, state.variables_critic_target, tau)

        return state.replace(
            variables_actor=new_variables_actor,
            variables_critic=new_variables_critic,
            variables_critic_target=new_variables_critic_target,
            opt_state_actor=new_opt_state_actor,
            opt_state_critic=new_opt_state_critic,
            opt_state_ent_coef=new_opt_state_ent_coef,
            log_ent_coef=new_log_ent_coef,
        ), XDSACTrainMetrics(
            loss_actor=loss_actor,
            loss_critic=loss_critic,
            loss_ent_coef=loss_ent_coef,
            grad_norm_actor=grad_norm_actor,
            grad_norm_critic=grad_norm_critic,
            mixture_entropy=jnp.mean(aux_mixture_entropy),
            mean_mixture_logits=aux_mean_mixture_logits,
            entropy_mean=jnp.mean(aux_entropy),
            entropy_std=jnp.sqrt(jnp.mean(jnp.square(aux_entropy - jnp.mean(aux_entropy)))),
            ent_coef=jnp.exp(new_log_ent_coef),
            kappa=kappa,
            q_val_min=jnp.min(target),
            q_val_max=jnp.max(target),
        )

    return XDSACImpl(
        init=init,
        act_deterministic=act_deterministic,
        act_stochastic=act_stochastic,
        train_step=train_step,
    )
