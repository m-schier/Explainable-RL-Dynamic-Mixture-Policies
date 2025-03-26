import chex
from typing import Callable, Optional, Tuple, Any, Union
from functools import partial
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import gymnasium as gym


def huber_loss(x, y, beta=1., where=None):
    chex.assert_equal_shape([x, y])

    loss = jax.lax.select(jnp.abs(x - y) < beta, .5 * jnp.square(x - y), jnp.abs(x - y) - .5 * beta)

    return jnp.mean(loss, where=where)


def mse_loss(x, y, where=None):
    chex.assert_equal_shape([x, y])
    return jnp.mean(jnp.square(x - y), where=where)


def get_loss(fn):
    if isinstance(fn, str):
        if fn in ('l2', 'mse'):
            return mse_loss
        elif fn == 'huber':
            return huber_loss
        else:
            raise ValueError(f"{fn = }")
    elif callable(fn):
        return fn
    else:
        raise TypeError(type(fn))


@jax.jit
def polyak_update(params, target_params, tau):
    return jax.tree_map(lambda p, tp: (1 - tau) * tp + tau * p, params, target_params)


@jax.jit
def default_batch_obs(obs):
    return obs[None, ...]


def sac_check_action_space(action_space: gym.Space):
    import numpy as np

    if not isinstance(action_space, gym.spaces.Box):
        raise TypeError(f"Unsupported action space type: {type(action_space)}")

    if len(action_space.shape) != 1:
        raise ValueError(f"Only 1D action spaces supported, shape was: {action_space.shape}")

    if not np.allclose(action_space.low, -1):
        raise ValueError(f"Bad lower bounds for action space: {action_space.low}")

    if not np.allclose(action_space.high, 1):
        raise ValueError(f"Bad upper bounds for action space: {action_space.high}")

    n_act, = action_space.shape
    return n_act


def sac_init_log_entropy_coefficient():
    return jnp.log(jnp.ones(tuple()) * .1)  # Initialize alpha=.1


def sac_split_fe_producers(fe_producer):
    if isinstance(fe_producer, tuple):
        critic_fe_producer, actor_fe_producer = fe_producer
    else:
        critic_fe_producer = fe_producer
        actor_fe_producer = fe_producer
    return critic_fe_producer, actor_fe_producer


def squashed_gaussian_sample(rng: jax.Array, mu: jax.Array, log_std: jax.Array):
    chex.assert_equal_shape([mu, log_std])
    return jnp.tanh(mu + jax.random.normal(rng, mu.shape) * jnp.exp(log_std))


def gaussian_log_prob(mu: jax.Array, log_std: jax.Array, values: jax.Array) -> jax.Array:
    # Based on torch.distributions.normal.Normal.log_prob
    chex.assert_equal_shape([mu, log_std, values])
    var = jnp.square(jnp.exp(log_std))
    log_prob = (
        -((values - mu) ** 2) / (2 * var)
        - log_std
        - jnp.log(jnp.sqrt(2 * jnp.pi))
    )
    # Sum over log_prob since continuous actions considered independent
    return jnp.sum(log_prob, axis=-1, keepdims=True)


def squashed_gaussian_sample_with_log_prob(rng: jax.Array, mu: jax.Array, log_std: jax.Array):
    chex.assert_equal_shape([mu, log_std])
    gaussian_actions = mu + jax.random.normal(rng, mu.shape) * jnp.exp(log_std)
    actions = jnp.tanh(gaussian_actions)
    log_prob = gaussian_log_prob(mu, log_std, gaussian_actions) - jnp.sum(jnp.log(1 - actions ** 2 + 1e-6), axis=-1,
                                                                          keepdims=True)
    return actions, log_prob


class SacCritic(nn.Module):
    fe_producer: Callable[[], nn.Module]
    net_arch: Tuple[int] = (256, 256)
    n_critics: int = 2
    share_feature_extractor: bool = True
    action_broadcast: bool = False

    @nn.compact
    def __call__(self, obs, actions, reduce=False):
        shared_obs_latent = self.fe_producer()(obs) if self.share_feature_extractor else None

        critic_vals = []

        for _ in range(self.n_critics):
            critic_obs_latent = self.fe_producer()(obs) if not self.share_feature_extractor else shared_obs_latent

            if self.action_broadcast:
                # If action broadcasting allowed, observations are expanded to fit extra action axis, that is:
                # Obs: REST       x F
                # Act: REST x ... x A
                if len(actions.shape) > len(critic_obs_latent.shape):
                    *batch_dims, feature_dim = critic_obs_latent.shape
                    chex.assert_equal(actions.shape[:len(batch_dims)], tuple(batch_dims))

                    extra_dim_count = len(actions.shape) - len(critic_obs_latent.shape)
                    critic_obs_latent = jnp.expand_dims(critic_obs_latent, range(-extra_dim_count - 1, -1))

                    # Concatenation below requires exact match on all other axes, not bc-compatible
                    critic_obs_latent = jnp.broadcast_to(critic_obs_latent, actions.shape[:-1] + (critic_obs_latent.shape[-1],))
                    # for i in range(-extra_dim_count - 1, -1):
                    #    critic_obs_latent = jnp.repeat(critic_obs_latent, actions.shape[i], axis=i)
            else:
                # Assert compatibility without broadcasting
                chex.assert_equal(critic_obs_latent.shape[:-1], actions.shape[:-1])

            x = jnp.concatenate([critic_obs_latent, actions], axis=-1)
            for dim in self.net_arch:
                x = nn.relu(nn.Dense(features=dim)(x))
            x = nn.Dense(features=1)(x)
            critic_vals.append(x)

        if reduce:
            return jax.tree_util.tree_reduce(lambda acc, x: jnp.minimum(acc, x), critic_vals)

        return critic_vals


class SacActor(nn.Module):
    fe_producer: Callable[[], nn.Module]
    net_arch: Tuple[int] = (256, 256)
    n_act: int = None

    def setup(self) -> None:
        self.feature_extrator = self.fe_producer()
        self.pi_layers = [nn.Dense(features=d) for d in self.net_arch]
        self.mu = nn.Dense(features=self.n_act)
        self.log_std = nn.Dense(features=self.n_act)

    def get_latent(self, x):
        x = self.feature_extrator(x)
        # chex.assert_rank(x, 2)

        for layer in self.pi_layers:
            x = nn.relu(layer(x))

        chex.assert_equal(x.shape[-1], self.net_arch[-1])
        # chex.assert_shape(x, (n, self.net_arch[-1]))
        return x

    def deterministic_action(self, x):
        return jnp.tanh(self.mu(self.get_latent(x)))

    def __call__(self, x, rng):
        """
        Return action mean and log std for given observations
        """
        latent = self.get_latent(x)

        rest_shape = latent.shape[:-1]
        # chex.assert_rank(latent, 2)

        mean = self.mu(latent)
        chex.assert_shape(mean, rest_shape + (self.n_act,))
        # Limits from SB3's SAC implementation
        log_std = jnp.clip(self.log_std(latent), -20, 2)
        chex.assert_equal_shape([mean, log_std])

        actions, log_prob = squashed_gaussian_sample_with_log_prob(rng, mean, log_std)
        chex.assert_equal_shape([mean, log_std, actions])
        chex.assert_shape(log_prob, actions.shape[:-1] + (1,))
        return actions, log_prob


@chex.dataclass(frozen=True)
class SoftActorCriticState:
    variables_actor: None
    variables_critic: None
    variables_critic_target: None
    log_ent_coef: None
    optimizer_state_critic: None
    optimizer_state_actor: None
    optimizer_state_ent_coef: None


@chex.dataclass(frozen=True)
class SoftActorCriticMetrics:
    loss_actor: float
    loss_critic: float
    loss_ent_coef: Optional[float]
    ent_coef: float
    entropy_mean: float
    q_val_min: float
    q_val_max: float
    grad_norm_actor: float
    grad_norm_critic: float


@chex.dataclass(frozen=True)
class SoftActorCriticImpl:
    init: Callable[[jax.Array, Any], SoftActorCriticState]
    train_step: Callable[[SoftActorCriticState, Any, jax.Array], Tuple[SoftActorCriticState, SoftActorCriticMetrics]]
    estimate_q: Callable[[SoftActorCriticState, Any, Any], jax.Array]
    act_deterministic: Callable[[SoftActorCriticState, Any], jax.Array]
    act_stochastic: Callable[[SoftActorCriticState, Any, jax.Array], jax.Array]


def make_sac(fe_producer, action_space, net_arch: Tuple[int] = (256, 256), n_critics: int = 2,
             ent_coef: Union[str, float] = "auto", tau: float = .005, gamma: float = .99, loss="huber",
             lr: float = 0.0003, q_lr: Optional[float] = None, critic_max_grad_norm=None,
             target_entropy: Optional[float] = None):
    n_act = sac_check_action_space(action_space)

    if target_entropy is None:
        target_entropy = -n_act

    critic_fe_producer, actor_fe_producer = sac_split_fe_producers(fe_producer)

    actor = SacActor(fe_producer=actor_fe_producer, net_arch=net_arch, n_act=n_act)
    critic = SacCritic(fe_producer=critic_fe_producer, net_arch=net_arch, n_critics=n_critics)
    loss_fn = get_loss(loss)

    q_lr = q_lr or lr

    optimizer_actor = optax.adam(learning_rate=lr)
    # Ent coeff lr could by higher, see
    # https://docs.cleanrl.dev/rl-algorithms/sac/#implementation-details
    optimizer_ent_coef = optax.adam(learning_rate=q_lr)

    if critic_max_grad_norm is None:
        optimizer_critic = optax.adam(learning_rate=q_lr)
    else:
        optimizer_critic = optax.chain(optax.clip_by_global_norm(critic_max_grad_norm), optax.adam(learning_rate=q_lr))

    def init(rng, dummy_obs) -> SoftActorCriticState:
        actor_rng, critic_rng = jax.random.split(rng)
        (dummy_actions, _), variables_actor = actor.init_with_output(actor_rng, dummy_obs, actor_rng)  # Use same rng for dummy call
        variables_critic = critic.init(critic_rng, dummy_obs, dummy_actions)
        variables_critic_target = jax.tree_map(lambda x: x.copy(), variables_critic)
        optimizer_state_actor = optimizer_actor.init(variables_actor['params'])
        optimizer_state_critic = optimizer_critic.init(variables_critic['params'])

        if ent_coef == "auto":
            log_ent_coef = sac_init_log_entropy_coefficient()
            optimizer_state_ent_coef = optimizer_ent_coef.init(log_ent_coef)
        else:
            log_ent_coef = jnp.log(ent_coef)
            optimizer_state_ent_coef = None

        return SoftActorCriticState(
            variables_actor=variables_actor,
            variables_critic=variables_critic,
            variables_critic_target=variables_critic_target,
            log_ent_coef=log_ent_coef,
            optimizer_state_actor=optimizer_state_actor,
            optimizer_state_critic=optimizer_state_critic,
            optimizer_state_ent_coef=optimizer_state_ent_coef,
        )

    @jax.jit
    def act_deterministic(state: SoftActorCriticState, obs: Any) -> jax.Array:
        return actor.apply(state.variables_actor, obs, method=SacActor.deterministic_action)

    @jax.jit
    def act_stochastic(state: SoftActorCriticState, obs: Any, rng: jax.Array):
        actions, _ = actor.apply(state.variables_actor, obs, rng)
        return actions

    @jax.jit
    def estimate_q(state: SoftActorCriticState, obs: Any, actions: Any):
        return critic.apply(state.variables_critic, obs, actions, reduce=True)

    @jax.jit
    def train_step(state: SoftActorCriticState, batch, rng: jax.Array):
        obs, next_obs, action, reward, done = batch
        next_act_rng, act_rng = jax.random.split(rng)

        next_actions, next_log_prob = actor.apply(state.variables_actor, next_obs, next_act_rng)
        # Next Q-Values min over all critics
        next_q_vals = critic.apply(state.variables_critic_target, next_obs, next_actions, reduce=True)

        # Entropy regularization
        next_q_vals = next_q_vals - jnp.exp(state.log_ent_coef) * next_log_prob
        target_q_vals = reward[..., None] + (1 - done)[..., None] * gamma * next_q_vals
        chex.assert_equal_shape([next_q_vals, next_log_prob, target_q_vals])

        c_vars, c_params_init = flax.core.pop(state.variables_critic, 'params')

        def critic_objective(p):
            current_q_vals = critic.apply({**c_vars, 'params': p}, obs, action)
            return .5 * sum(loss_fn(q, target_q_vals) for q in current_q_vals)

        loss_critic, grads = jax.value_and_grad(critic_objective)(c_params_init)
        grad_norm_critic = optax.global_norm(grads)
        updates, new_optimizer_state_critic = optimizer_critic.update(grads, state.optimizer_state_critic, c_params_init)
        new_variables_critic = {**c_vars, 'params': optax.apply_updates(c_params_init, updates)}

        a_vars, a_params_init = flax.core.pop(state.variables_actor, 'params')

        def actor_objective(p):
            actions_pi, log_prob = actor.apply({**a_vars, 'params': p}, obs, act_rng)
            min_qf_pi = critic.apply(new_variables_critic, obs, actions_pi, reduce=True)
            chex.assert_equal_shape([log_prob, min_qf_pi])
            return jnp.mean(jnp.exp(state.log_ent_coef) * log_prob - min_qf_pi), log_prob

        (loss_actor, aux_log_prob), grads = jax.value_and_grad(actor_objective, has_aux=True)(a_params_init)
        grad_norm_actor = optax.global_norm(grads)
        updates, new_optimizer_state_actor = optimizer_actor.update(grads, state.optimizer_state_actor, a_params_init)
        new_variables_actor = {**a_vars, 'params': optax.apply_updates(a_params_init, updates)}

        # Optionally update entropy
        def ent_coef_objective(p):
            return -jnp.mean(p * (aux_log_prob + target_entropy))

        if ent_coef == "auto":
            loss_ent_coef, grads = jax.value_and_grad(ent_coef_objective)(state.log_ent_coef)
            updates, new_optimizer_state_ent_coef = optimizer_ent_coef.update(grads, state.optimizer_state_ent_coef, state.log_ent_coef)
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
            log_ent_coef=new_log_ent_coef,
            optimizer_state_actor=new_optimizer_state_actor,
            optimizer_state_critic=new_optimizer_state_critic,
            optimizer_state_ent_coef=new_optimizer_state_ent_coef
        ), SoftActorCriticMetrics(
            loss_actor=loss_actor,
            loss_critic=loss_critic,
            loss_ent_coef=loss_ent_coef,
            grad_norm_actor=grad_norm_actor,
            grad_norm_critic=grad_norm_critic,
            q_val_min=jnp.min(target_q_vals),
            q_val_max=jnp.max(target_q_vals),
            ent_coef=jnp.exp(new_log_ent_coef),
            entropy_mean=jnp.mean(-aux_log_prob),  # log probs are sampled according to pdf so just avg for entropy
        )

    return SoftActorCriticImpl(
        init=init,
        train_step=train_step,
        act_deterministic=act_deterministic,
        act_stochastic=act_stochastic,
        estimate_q=estimate_q,
    )
