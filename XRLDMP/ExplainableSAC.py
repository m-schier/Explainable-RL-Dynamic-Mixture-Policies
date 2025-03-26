import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import optax
import chex
import gymnasium as gym

from functools import partial
from typing import Callable, Tuple, Optional
from .SAC import get_loss, polyak_update, SacCritic, sac_check_action_space, sac_init_log_entropy_coefficient
from .ExplainableDSAC import sample_categorical_probs


def sample_squashed_gaussian_mixture(rng: jax.Array, weights: jax.Array, mu: jax.Array, log_std: jax.Array, return_gaussian_actions=False):
    """
    Sample from a squashed diagonal gaussian mixture model. This method is NOT differentiable!
    :param rng: RNG state
    :param weights: Mixture weights (probabilities not logits!) in shape ... x K x 1
    :param mu: Means of mixture components in shape ... x K x A
    :param log_std: Logs of standard deviation of mixture components in shape ... x K x A
    :return: Sampled actions in shape ... x A
    """
    mixture_rng, std_rng = jax.random.split(rng)

    chex.assert_equal_shape([mu, log_std])
    chex.assert_equal(weights.shape, mu.shape[:-1] + (1,))

    weights = jnp.squeeze(weights, -1)

    which = sample_categorical_probs(mixture_rng, weights)
    chex.assert_equal(which.shape, weights.shape[:-1])

    selected_mu = jnp.squeeze(jnp.take_along_axis(mu, which[..., None, None], axis=-2), (-2))
    chex.assert_equal(selected_mu.shape, which.shape + (mu.shape[-1],))
    selected_log_std = jnp.squeeze(jnp.take_along_axis(log_std, which[..., None, None], axis=-2), (-2))

    gaussian_actions = selected_mu + jax.random.normal(std_rng, selected_mu.shape) * jnp.exp(selected_log_std)

    if return_gaussian_actions:
        return jnp.tanh(gaussian_actions), gaussian_actions
    else:
        return jnp.tanh(gaussian_actions)


def components_pdf(mu: jax.Array, log_std: jax.Array, values: jax.Array, component_count=1):
    """
    :param mu: Means of mixture components in shape ... x K x A
    :param log_std: Logs of standard deviation of mixture components in shape ... x K x A
    :param values: Actions in shape ... x A
    :return: components prob in shape ... x 1
    """
    chex.assert_equal_shape([mu, log_std])
    chex.assert_equal(values.shape, mu.shape[:-component_count - 1] + mu.shape[-1:])
    var = jnp.square(jnp.exp(log_std))

    values = jnp.expand_dims(values, range(-component_count - 1, -1))

    pdf = 1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-.5 * jnp.square(values - mu) / var)
    # Prod over log_prob since continuous actions considered independent
    return jnp.prod(pdf, axis=-1, keepdims=True)


def squashed_gmm_log_pdf(weights, mu, log_std, unsquashed_vals):
    """
    :param weights: Mixture weights (probabilities not logits!) in shape ... x K x 1
    :param mu: Means of mixture components in shape ... x K x A
    :param log_std: Logs of standard deviation of mixture components in shape ... x K x A
    :param unsquashed_vals: Unsquashed actions in shape ... x ,,, x A
    :return: log pdf in shape ... x ,,, x 1
    """

    chex.assert_equal_shape([mu, log_std])

    *batch_dim, components_dim, action_dim = mu.shape

    # Everything else is currently an error, but could be supported
    chex.assert_equal(len(batch_dim), 1)
    batch_dim = tuple(batch_dim)

    chex.assert_equal(batch_dim, unsquashed_vals.shape[:len(batch_dim)])
    chex.assert_equal(unsquashed_vals.shape[-1], action_dim)

    comma_len = len(unsquashed_vals.shape) - len(batch_dim) - 1

    # ... x ,,, x A, not broadcast
    squashed_vals = jnp.tanh(unsquashed_vals)

    # Now ... x ,,, x 1 x A
    unsquashed_vals = unsquashed_vals[..., None, :]

    mu = jnp.expand_dims(mu, range(-comma_len - 2, -2))
    log_std = jnp.expand_dims(log_std, range(-comma_len - 2, -2))
    weights = jnp.expand_dims(weights, range(-comma_len - 2, -2))

    var = jnp.square(jnp.exp(log_std))
    component_pdf = 1 / jnp.sqrt(2 * jnp.pi * var) * jnp.exp(-.5 * jnp.square(unsquashed_vals - mu) / var)
    component_pdf = jnp.prod(component_pdf, axis=-1, keepdims=True)  # Multiply independent action dims

    # Then ... x ,,, x 1
    gmm_pdf = jnp.sum(weights * component_pdf, axis=-2)

    # Change of variables (See "Soft Actor-Critic", Appendix C). Clipping from SB3 default.
    log_divisor = jnp.sum(jnp.log(1 - squashed_vals ** 2 + 1e-6), axis=-1, keepdims=True)
    chex.assert_equal_shape([gmm_pdf, log_divisor])

    result = jnp.log(jnp.maximum(gmm_pdf, 1e-10)) - log_divisor
    return result


def sample_squashed_gaussian_mixture_with_log_prob(rng: jax.Array, weights: jax.Array, mu: jax.Array, log_std: jax.Array):
    actions, gaussian_actions = sample_squashed_gaussian_mixture(rng, weights, mu, log_std, return_gaussian_actions=True)

    component_pdf = components_pdf(mu, log_std, gaussian_actions)
    chex.assert_equal(component_pdf.shape, weights.shape)

    unsquashed_pdf = jnp.sum(jnp.squeeze(weights * component_pdf, -1), -1, keepdims=True)
    chex.assert_equal(unsquashed_pdf.shape, weights.shape[:-2] + (1,))

    # Change of variables (See "Soft Actor-Critic", Appendix C)
    divisor = jnp.prod(1 - actions ** 2 + 1e-6, axis=-1, keepdims=True)
    chex.assert_equal_shape([unsquashed_pdf, divisor])

    log_pdf = jnp.log(unsquashed_pdf / divisor)

    return actions, log_pdf


def squashed_gm_pdf(unsquashed_xs, phis, mus, sigmas):
    """
    Calculate PDF of a squashed Gaussian mixture distribution
    :param unsquashed_xs: Query points in shape ... x P x A without squashing!
    :param phis: Component weights in shape ... x K x 1
    :param mus: Component mus in shape ... x K x A
    :param sigmas: Component sigmas in shape ... x K x A
    :return: PDF in shape ... x P
    """
    chex.assert_equal_shape([mus, sigmas])
    chex.assert_equal(mus.shape[:-1] + (1,), phis.shape)

    *batch_dims, n_components, n_act = mus.shape
    batch_dims = tuple(batch_dims)

    *check_dims, n_points, _ = unsquashed_xs.shape
    chex.assert_equal(tuple(check_dims), batch_dims)
    chex.assert_equal(n_act, unsquashed_xs.shape[-1])

    var = jnp.square(sigmas)

    squashed_xs = jnp.tanh(unsquashed_xs)

    # ... x P x K x A
    component_pdf = 1 / jnp.sqrt(2 * jnp.pi * var[..., None, :, :]) * jnp.exp(
        -.5 * jnp.square(unsquashed_xs[..., None, :] - mus[..., None, :, :]) / var[..., None, :, :])
    chex.assert_equal(component_pdf.shape, (*batch_dims, n_points, n_components, n_act))
    # ... x P x K
    component_pdf = jnp.prod(component_pdf, axis=-1)  # Product over independent action dims
    chex.assert_equal(component_pdf.shape, (*batch_dims, n_points, n_components))

    # ... x P
    phis = jnp.squeeze(phis, -1)
    pdf = jnp.sum(phis[..., None, :] * component_pdf, axis=-1)
    chex.assert_equal(pdf.shape, (*batch_dims, n_points))

    # ... x OC
    divisor = jnp.prod(1 - squashed_xs ** 2 + 1e-6, axis=-1)
    pdf = pdf / divisor
    chex.assert_equal(pdf.shape, (*batch_dims, n_points))

    return pdf, squashed_xs


def squashed_gm_approx_max(phis, mus, sigmas):
    """
    Approximate the maximum of batched squashed Gaussian mixture
    :param phis: Component weights in shape ... x K x 1
    :param mus: Component mus in shape ... x K x A
    :param sigmas: Component sigmas in shape ... x K x A
    :return: Maximum in shape ... x A
    """
    chex.assert_equal_shape([mus, sigmas])
    chex.assert_equal(mus.shape[:-1] + (1,), phis.shape)

    *batch_dims, n_components, n_act = mus.shape
    batch_dims = tuple(batch_dims)

    unsquashed_acts = mus

    pdf, squashed_acts = squashed_gm_pdf(unsquashed_acts, phis, mus, sigmas)

    chex.assert_equal(pdf.shape, (*batch_dims, n_components))

    indices = jnp.argmax(pdf, axis=-1)

    result = jnp.take_along_axis(squashed_acts, indices[..., None, None], axis=-2)
    result = jnp.squeeze(result, -2)
    chex.assert_equal(result.shape, batch_dims + (n_act,))

    return result


def _sample_no_grad(rng: jax.Array, act_result):
    """
    Fast sampling that is not differentiable through the mixture component
    """
    # Mus and sigmas have an extra component dimension, must flatten and repeat weights accordingly
    *rest, object_dim, component_dim, act_dim = act_result['object_means'].shape

    mus = act_result['object_means'].reshape(*rest, object_dim * component_dim, act_dim)
    log_stds = act_result['object_log_stds'].reshape(*rest, object_dim * component_dim, act_dim)
    phis = act_result['weights'].reshape(*rest, object_dim * component_dim, 1)

    return sample_squashed_gaussian_mixture_with_log_prob(rng, phis, mus, log_stds)


class GaussianMixturePolicy(nn.Module):
    n_act: int
    observation_shaper: Callable
    uniform_mixture: bool = False
    net_arch: Tuple[int] = (256, 256, 256, 256)
    components_per_object: int = 1
    weight_latent_depth: int = -1

    def setup(self) -> None:
        self.hidden_layers = [nn.Dense(features=f) for f in self.net_arch]
        self.phi = None if self.uniform_mixture else nn.Dense(features=1)
        self.mu = nn.Dense(features=self.n_act * self.components_per_object)
        self.log_std = nn.Dense(features=self.n_act * self.components_per_object)

    def _get_latent_depth(self):
        phi_latent_depth = self.weight_latent_depth

        if phi_latent_depth < 0:
            phi_latent_depth = len(self.hidden_layers) + phi_latent_depth

        if phi_latent_depth < 0 or phi_latent_depth >= len(self.hidden_layers):
            raise AssertionError(f"Invalid weight branch depth {self.weight_latent_depth = }, {len(self.hidden_layers) = }")

        return phi_latent_depth

    def _intermediate(self, inp):
        x, present = self.observation_shaper(inp)

        *rest, object_dim, input_dim = x.shape
        chex.assert_axis_dimension_gt(x, -2, 0)  # Object dim of 0 is never supported
        # chex.assert_rank(x, 3)  # Should be batch_size x num_objects x features
        chex.assert_equal(present.shape, x.shape[:-1] + (1,))

        if self.phi is None:
            mixture_weight_logits = jnp.zeros_like(x[..., :1])
        else:
            phi_latent_depth = self._get_latent_depth()

            for layer in self.hidden_layers[:phi_latent_depth + 1]:
                x = layer(x)
                x = nn.relu(x)

            mixture_weight_logits = self.phi(x)
            mixture_weight_logits = mixture_weight_logits - jnp.mean(mixture_weight_logits, axis=-2, keepdims=True)

        chex.assert_equal_shape([mixture_weight_logits, present])
        return x, present, mixture_weight_logits

    def _final_latent(self, x):
        if self.phi is None:
            head_start = 0
        else:
            head_start = self._get_latent_depth() + 1

        for layer in self.hidden_layers[head_start:]:
            x = layer(x)
            x = nn.relu(x)
        return x

    def act_fast(self, inp, rng):
        object_rng, component_rng, normal_rng = jax.random.split(rng, 3)

        x, present, mixture_weight_logits = self._intermediate(inp)
        print(f"{x.shape = }, {present.shape = }")
        *rest, object_dim, _ = x.shape

        mixture_weight_logits = jnp.where(present, mixture_weight_logits, jnp.full_like(mixture_weight_logits, -1e20))
        print(f"{mixture_weight_logits.shape = }")
        selected_components = jnp.expand_dims(jax.random.categorical(object_rng, mixture_weight_logits, -2), -2)
        print(f"{selected_components.shape = }")

        x = jnp.take_along_axis(x, selected_components, -2)

        x = self._final_latent(x)

        mean = self.mu(x).reshape(*rest, 1, self.components_per_object, self.n_act)
        # SB3 has -20 as lower limit for log std, CleanRL -5
        log_std = jnp.clip(self.log_std(x), -5, 2).reshape(*rest, 1, self.components_per_object, self.n_act)

        # Currently uniform component weight per object, which makes things easy, we can sample one

        chex.assert_axis_dimension(mean, -3, 1)

        selected_objects = jax.random.randint(component_rng, tuple(rest) + (1, 1, 1), minval=0, maxval=self.components_per_object)

        mean = jnp.take_along_axis(mean, selected_objects, axis=-2)
        log_std = jnp.take_along_axis(log_std, selected_objects, axis=-2)

        mean = jnp.squeeze(mean, (-3, -2))
        log_std = jnp.squeeze(log_std, (-3, -2))

        gaussian_actions = mean + jnp.exp(log_std) * jax.random.normal(normal_rng, log_std.shape)
        chex.assert_shape(gaussian_actions, tuple(rest) + (self.n_act,))
        return jnp.tanh(gaussian_actions)

    def __call__(self, inp):
        x, present, mixture_weight_logits = self._intermediate(inp)
        *rest, object_dim, _ = x.shape

        x = self._final_latent(x)

        mixture_weights = jax.nn.softmax(mixture_weight_logits, axis=-2, where=present, initial=-jnp.inf)

        # If no object is given (which is not allowed!), fall back to safe behavior
        valid_mixture = jnp.sum(mixture_weights, axis=-2, keepdims=True) > 0.
        mixture_weights = jax.lax.select(jnp.broadcast_to(valid_mixture, mixture_weights.shape), mixture_weights,
                                         jnp.ones_like(mixture_weights) / mixture_weights.shape[-2])

        mean = self.mu(x).reshape(*rest, object_dim, self.components_per_object, self.n_act)
        # SB3 has -20 as lower limit for log std, CleanRL -5
        log_std = jnp.clip(self.log_std(x), -5, 2).reshape(*rest, object_dim, self.components_per_object, self.n_act)

        # Mixture weights are currently uniform per object, broadcast to components per object and normalize to keep sum
        mixture_weights = jnp.broadcast_to(mixture_weights[..., None, :], mean.shape[:-1] + (1,)) / self.components_per_object

        return {
            'object_means': mean,  # mus in shape ... O x C x A
            'object_log_stds': log_std,  # sigmas in shape ... O x C x A
            'weights': mixture_weights,  # phis in shape ... x O x C x 1
        }


@chex.dataclass(frozen=True)
class CMMSACTrainMetrics:
    loss_actor: None
    loss_critic: None
    loss_ent_coef: Optional[float]
    grad_norm_actor: None
    grad_norm_critic: None
    mixture_entropy: None
    entropy_mean: None
    entropy_std: None
    ent_coef: None
    kappa: None


@chex.dataclass(frozen=True)
class XSACState:
    variables_actor: None
    variables_critic: None
    variables_critic_target: None
    opt_state_actor: None
    opt_state_critic: None
    opt_state_ent_coef: None
    log_ent_coef: None


@chex.dataclass(frozen=True)
class XSACImpl:
    init: None
    act_deterministic: None
    act_stochastic: None
    train_step: None


def make_xsac(feature_reshaper_actor: Callable, fe_producer_critic: Callable, action_space: gym.Space,
              actor_net_arch: Tuple[int] = (256, 256), critic_net_arch: Tuple[int] = (256, 256),
              n_critics=2, ent_coef="auto", lr: float = 1e-4, q_lr: Optional[float] = None, gamma: float = .99,
              loss='huber', tau: float = 5e-3,
              uniform_mixture: bool = False, kappa: float = .01, clip_norm: Optional[float] = None,
              weight_latent_depth: int = -1,
              target_entropy: Optional[float] = None, components_per_object: int = 1):
    """
    A spin on an explainable DSAC on dynamic observation spaces using myopic actors.
    :param feature_reshaper_actor: The (usually not learnable) function that reshapes the observation for the
    myopic per-object actors.
    :param fe_producer_critic: Feature extractor producer
    :param action_space: Action space of the environment, must be discrete
    :param actor_net_arch: Net architectures in units per layer for the actor
    :param critic_net_arch: Net architectures in units per layer for the critics
    :param n_critics: Number of critics, 2 in standard SAC and paper
    :param ent_coef: Weighting of the entropy term in the objective
    :param lr: Learning rate for both actor and critic
    :param gamma: Discount factor
    :param loss: Loss for critic. Paper uses L2, huber sometimes considered more stable. Actor uses gradient ascent.
    :param tau: Step size for critic target update, default from SB3 SAC, SAC-Discrete paper uses hard updates
    :return: Implementation of DSAC
    """
    n_act = sac_check_action_space(action_space)

    if target_entropy is None:
        target_entropy = -n_act  # Only applies for automatic entropy

    critic = SacCritic(fe_producer=fe_producer_critic, net_arch=critic_net_arch, n_critics=n_critics,
                       action_broadcast=True)
    actor = GaussianMixturePolicy(n_act=n_act, observation_shaper=feature_reshaper_actor, net_arch=actor_net_arch,
                                  uniform_mixture=uniform_mixture, components_per_object=components_per_object,
                                  weight_latent_depth=weight_latent_depth)
    loss_fn = get_loss(loss)
    q_lr = q_lr or lr

    if clip_norm is None:
        optimizer_actor = optax.adam(learning_rate=lr)
        optimizer_critic = optax.adam(learning_rate=q_lr)
    else:
        optimizer_actor = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=lr))
        optimizer_critic = optax.chain(optax.clip_by_global_norm(clip_norm), optax.adam(learning_rate=q_lr))

    optimizer_ent_coef = optax.adam(learning_rate=q_lr)

    def init(rng: jax.Array, obs) -> XSACState:
        dummy_act_result, variables_actor = actor.init_with_output(rng, obs)
        dummy_actions, _ = _sample_no_grad(rng, dummy_act_result)  # Reuse rng, result content does not matter
        variables_critic = critic.init(rng, obs, dummy_actions)
        variables_critic_target = jax.tree_map(lambda x: x.copy(), variables_critic)
        opt_state_actor = optimizer_actor.init(variables_actor['params'])
        opt_state_critic = optimizer_critic.init(variables_critic['params'])

        if ent_coef == "auto":
            log_ent_coef = sac_init_log_entropy_coefficient()
            opt_state_ent_coef = optimizer_ent_coef.init(log_ent_coef)
        else:
            log_ent_coef = jnp.log(ent_coef)
            opt_state_ent_coef = None

        return XSACState(
            variables_actor=variables_actor,
            variables_critic=variables_critic,
            variables_critic_target=variables_critic_target,
            opt_state_actor=opt_state_actor,
            opt_state_critic=opt_state_critic,
            opt_state_ent_coef=opt_state_ent_coef,
            log_ent_coef=log_ent_coef,
        )

    @partial(jax.jit, static_argnums=(3,))
    def act_stochastic(state: XSACState, obs, rng, return_full=False):
        if return_full:
            act_result = actor.apply(state.variables_actor, obs)
            choice, _ = _sample_no_grad(rng, act_result)

            return choice, act_result
        else:
            return actor.apply(state.variables_actor, obs, rng, method=actor.act_fast)

    @partial(jax.jit, static_argnums=(2,))
    def act_deterministic(state: XSACState, obs, return_full=False):
        act_result = actor.apply(state.variables_actor, obs)

        # We do not support deterministic acting on XSAC as finding the argmax is nontrivial, however this function
        # is provided to extract the mixture parameterization. In the future we should probably change the interface.

        if return_full:
            return None, act_result
        else:
            return None

    @jax.jit
    def train_step(state: XSACState, batch, rng) -> Tuple[XSACState, CMMSACTrainMetrics]:
        obs, next_obs, action, reward, done = batch
        n, = done.shape

        next_act_rng, act_rng = jax.random.split(rng)

        next_act_result = actor.apply(state.variables_actor, next_obs)

        next_actions, next_log_prob = _sample_no_grad(next_act_rng, next_act_result)
        chex.assert_equal(next_actions.shape, (n, n_act))
        chex.assert_equal(next_log_prob.shape, (n, 1))

        # Next Q-Values min over all critics
        next_q_vals = critic.apply(state.variables_critic_target, next_obs, next_actions, reduce=True)
        chex.assert_equal(next_q_vals.shape, (n, 1))

        # Entropy regularization
        next_q_vals = next_q_vals - jnp.exp(state.log_ent_coef) * next_log_prob
        target_q_vals = reward[..., None] + (1 - done)[..., None] * gamma * next_q_vals
        chex.assert_equal_shape([next_q_vals, next_log_prob, target_q_vals])

        c_vars, c_params_init = flax.core.pop(state.variables_critic, 'params')

        def critic_objective(p):
            current_q_vals = critic.apply({**c_vars, 'params': p}, obs, action)
            return .5 * sum(loss_fn(q, target_q_vals) for q in current_q_vals)

        loss_critic, grads_critic = jax.value_and_grad(critic_objective)(c_params_init)
        grad_norm_critic = optax.global_norm(grads_critic)
        updates, new_opt_state_critic = optimizer_critic.update(grads_critic, state.opt_state_critic, c_params_init)
        new_variables_critic = {**c_vars, 'params': optax.apply_updates(c_params_init, updates)}

        a_vars, a_params_init = flax.core.pop(state.variables_actor, 'params')

        def actor_objective(p):
            act_result = actor.apply({**a_vars, 'params': p}, obs)
            weights, mus, log_stds = act_result['weights'], act_result['object_means'], act_result['object_log_stds']
            # Weights in ... x O x C x 1
            # Others  in ... x O x C x A
            *batch_dims, object_dims, component_dims, act_dims = mus.shape

            uc_actions = mus + jax.random.normal(act_rng, mus.shape) * jnp.exp(log_stds)
            sc_actions = jnp.tanh(uc_actions)

            min_qf_pi = critic.apply(new_variables_critic, obs, sc_actions, reduce=True)
            chex.assert_shape(min_qf_pi, tuple(batch_dims) + (object_dims, component_dims, 1))

            # Reduce over object and component axes
            chex.assert_equal_shape([weights, min_qf_pi])
            min_qf_pi = jnp.sum(weights * min_qf_pi, axis=(-3, -2))

            # Calculate action entropy per component
            log_prob = squashed_gmm_log_pdf(
                weights.reshape(*batch_dims, object_dims * component_dims, 1),
                mus.reshape(*batch_dims, object_dims * component_dims, act_dims),
                log_stds.reshape(*batch_dims, object_dims * component_dims, act_dims),
                uc_actions
            )

            # Then take into account categorical
            log_prob = jnp.sum(weights * log_prob, axis=(-3, -2))

            chex.assert_equal_shape([log_prob, min_qf_pi])

            mix_probs = weights.reshape(*weights.shape[:-3], -1)  # ... x (O * C)
            mixture_entropy = -jnp.sum(jnp.log(jnp.clip(mix_probs, 1e-10, 1)) * mix_probs, axis=-1)
            chex.assert_equal(mixture_entropy.shape, (n,))

            return jnp.mean(jnp.exp(state.log_ent_coef) * log_prob - min_qf_pi - kappa * mixture_entropy), (log_prob, mixture_entropy)

        (loss_actor, (aux_log_prob, aux_mixture_entropy)), grads_actor = jax.value_and_grad(actor_objective, has_aux=True)(a_params_init)
        grad_norm_actor = optax.global_norm(grads_actor)
        updates, new_opt_state_actor = optimizer_actor.update(grads_actor, state.opt_state_actor, a_params_init)
        new_variables_actor = {**a_vars, 'params': optax.apply_updates(a_params_init, updates)}

        # Optionally update entropy
        def ent_coef_objective(p):
            return -jnp.mean(p * (aux_log_prob + target_entropy))

        if ent_coef == "auto":
            loss_ent_coef, grads = jax.value_and_grad(ent_coef_objective)(state.log_ent_coef)
            updates, new_opt_state_ent_coef = optimizer_ent_coef.update(grads, state.opt_state_ent_coef, state.log_ent_coef)
            new_log_ent_coef = optax.apply_updates(state.log_ent_coef, updates)
        else:
            loss_ent_coef = None
            new_opt_state_ent_coef = state.opt_state_ent_coef
            new_log_ent_coef = state.log_ent_coef

        # Polyak update
        new_variables_critic_target = polyak_update(new_variables_critic, state.variables_critic_target, tau)

        return state.replace(
            variables_actor=new_variables_actor,
            variables_critic=new_variables_critic,
            variables_critic_target=new_variables_critic_target,
            opt_state_actor=new_opt_state_actor,
            opt_state_critic=new_opt_state_critic,
            opt_state_ent_coef=new_opt_state_ent_coef,
            log_ent_coef=new_log_ent_coef,
        ), CMMSACTrainMetrics(
            loss_actor=loss_actor,
            loss_critic=loss_critic,
            loss_ent_coef=loss_ent_coef,
            grad_norm_actor=grad_norm_actor,
            grad_norm_critic=grad_norm_critic,
            mixture_entropy=jnp.mean(aux_mixture_entropy),
            entropy_mean=jnp.mean(-aux_log_prob),
            entropy_std=jnp.sqrt(jnp.mean(jnp.square(aux_log_prob - jnp.mean(aux_log_prob)))),
            ent_coef=jnp.exp(new_log_ent_coef),
            kappa=kappa,
        )

    return XSACImpl(
        init=init,
        act_deterministic=act_deterministic,
        act_stochastic=act_stochastic,
        train_step=train_step,
    )
