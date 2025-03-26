import sys

import cairo
import numpy as np


def draw_discrete_explanation_axis(ctx, device_pos, weight, probs, s=10):
    from CarEnv.Rendering.Rendering import stroke_fill

    long_order = [2, 1, 0]

    long_colors = np.array([
        [1., 0., 0.],
        [1., 1., 1.],
        [0., 1., 0.],
    ])

    dx, dy = device_pos
    ctx.save()
    ctx.identity_matrix()
    ctx.translate(dx, dy)
    ctx.scale(weight, weight)

    probs = probs.reshape(3, 3)

    for i in range(3):
        for j in range(3):
            ctx.rectangle((i - 1.5) * s, (long_order[j] - 2.) * s, s, s)
            c = long_colors[j] * probs[i, j]
            stroke_fill(ctx, (0., 0., 0.), c)

    ctx.restore()


def draw_discrete_explanation_pedals(ctx, device_pos, weight, probs, s=10):
    from CarEnv.Rendering.Rendering import stroke_fill

    long_order = [1, 3, 0, 2]  # Where 0 is top row

    long_colors = np.array([
        [1., 1., 1.],  # Neutral
        [1., 0., 0.],  # Brake
        [0., 1., 0.],  # Accelerate
        [1., .6, 0.],  # Accelerate+Brake
    ])

    dx, dy = device_pos
    ctx.save()
    ctx.identity_matrix()
    ctx.translate(dx, dy)
    ctx.scale(weight, weight)

    probs = probs.reshape(3, 4)

    for i in range(3):
        for j in range(4):
            ctx.rectangle((i - 1.5) * s, (long_order[j] - 2.) * s, s, s)
            c = long_colors[j] * probs[i, j]
            stroke_fill(ctx, (0., 0., 0.), c)

    ctx.restore()


def draw_discrete_explanation(ctx, device_pos, weight, probs, s=10, arrow_pt=None):
    if weight <= .01:
        return

    if arrow_pt is not None:
        ctx.save()
        ctx.identity_matrix()
        ctx.move_to(*arrow_pt)
        ctx.line_to(*device_pos)
        ctx.set_source_rgb(0., 0., 0.)
        ctx.set_line_width(1.)
        ctx.stroke()
        ctx.restore()

    if probs.shape == (12,):
        return draw_discrete_explanation_pedals(ctx, device_pos, weight, probs, s=s)
    elif probs.shape == (9,):
        return draw_discrete_explanation_axis(ctx, device_pos, weight, probs, s=s)
    else:
        raise ValueError(f"{probs.shape = }")


class SingleDiscreteExplainableRenderCallback:
    def __init__(self):
        self.policy = None

    def __call__(self, renderer, env, ctx: cairo.Context):
        if self.policy is None:
            return

        # Despite the name, this is up to date
        sensor_obs = env.last_observations['cones_set']

        # Filter present
        sensor_obs = sensor_obs[sensor_obs[..., 0] > 0]

        vn = env.sensors['cones_set'].view_normalizer

        trans = np.linalg.inv(env.ego_transform)

        policy_explanation = self.policy(env.last_observations)
        object_probs = policy_explanation['object_probs']
        weights = policy_explanation.get('weights', np.ones_like(object_probs[..., :1]))
        weights = np.squeeze(weights, axis=-1)

        # Always stretch probs to 100% to improve readability of visualization
        object_probs = object_probs / np.max(object_probs)
        weights = weights / np.max(weights)

        for (present, x, y, blue, yellow), weight, probs in zip(sensor_obs, weights, object_probs):
            if np.isclose(weight, 0):
                continue

            global_pos = trans @ np.array([x * vn, y * vn, 1.])

            dx, dy = ctx.user_to_device(global_pos[0], global_pos[1])
            # s = 10
            ex_dx = dx + yellow * 45 - blue * 45
            draw_discrete_explanation(ctx, (ex_dx, dy), weight, probs, arrow_pt=(dx, dy))

        draw_categorical_total_expl(ctx, policy_explanation)


class TrackingExplainableRenderCallback:
    def __init__(self):
        self.policy = None

    def __call__(self, renderer, env, ctx: cairo.Context):
        from .Wrappers import CarEnvTrajectoryToSetWrapper

        if self.policy is None:
            return

        # Despite the name, this is up to date
        last_obs = CarEnvTrajectoryToSetWrapper.set_transform(env.last_observations)
        sensor_obs = last_obs['trajectory_set']

        # Filter present
        sensor_obs = sensor_obs[sensor_obs[..., 0] > 0]

        vn = env.sensors['trajectory'].normalizer

        trans = np.linalg.inv(env.ego_transform)

        policy_explanation = self.policy(last_obs)
        object_probs = policy_explanation['object_probs']
        weights = policy_explanation.get('weights', np.ones_like(object_probs[..., :1]))
        weights = np.squeeze(weights, axis=-1)

        # Always stretch probs to 100% to improve readability of visualization
        object_probs = object_probs / np.max(object_probs)
        weights = weights / np.max(weights)

        for i, ((present, index, x, y), weight, probs) in enumerate(zip(sensor_obs, weights, object_probs)):
            if np.isclose(weight, 0):
                continue

            global_pos = trans @ np.array([x * vn, y * vn, 1.])

            dx, dy = ctx.user_to_device(global_pos[0], global_pos[1])
            # s = 10
            ex_dx = dx + 45 * (1 if i % 2 == 0 else -1)
            draw_discrete_explanation(ctx, (ex_dx, dy), weight, probs, arrow_pt=(dx, dy))

        draw_categorical_total_expl(ctx, policy_explanation)


class MARLExplainableRenderCallback:
    def __init__(self, discrete=True):
        self.policy = None
        self.tracked_agent = None
        self._last_tracked = set()
        self.discrete = discrete

    def __call__(self, renderer, env, ctx: cairo.Context):
        from CarEnv.Rendering.Rendering import stroke_fill
        from CarEnv.Util import inverse_transform_from_pose

        if self.tracked_agent not in env.agents:
            self.tracked_agent = None

        if self.tracked_agent is None:
            for ag in env.agents:
                if ag not in self._last_tracked:
                    self.tracked_agent = ag
                    break
            else:
                self._last_tracked = set(env.agents)
                return

        self._last_tracked = set(env.agents)

        sensors = env.get_agent(self.tracked_agent).sensors
        pose = env.objects[self.tracked_agent].model.pose
        trans = inverse_transform_from_pose(pose)

        ctx.arc(*pose[:2], 2., 0., 2 * np.pi)
        stroke_fill(ctx, (0., 0., 0.), None)

        obs = {k: v.last_observation for k, v in sensors.items()}

        if self.policy is None:
            return

        policy_explanation = self.policy({k: v[None] for k, v in obs.items()})

        if self.discrete:
            weights = policy_explanation['weights']
            object_probs = policy_explanation['object_probs']

            weights = np.squeeze(weights, (0, 2))
            weights = weights / np.max(weights)
            object_probs = np.squeeze(object_probs, 0)

            assert weights.shape == (len(obs['vehicles_set']) + 1,)

            args = [weights, object_probs]
            draw_fn = draw_discrete_explanation
        else:
            object_means = np.squeeze(np.array(policy_explanation['object_means']), 0)
            object_stds = np.squeeze(np.exp(np.array(policy_explanation['object_log_stds'])), 0)
            weights = np.squeeze(np.array(policy_explanation['weights']), 0)
            # Normalize by adding components, thus max is 1 for all components of an object
            weights = weights / np.max(np.sum(weights, axis=-2))
            args = [weights, object_means, object_stds]
            draw_fn = draw_diag_gm_explanation

        # Draw zero action
        dx, dy = ctx.user_to_device(*pose[:2])
        draw_fn(ctx, (dx + 20., dy + 20.), *[arg[0] for arg in args], arrow_pt=(dx, dy))

        vn = sensors['vehicles_set'].view_normalizer

        # Draw other actions
        for (present, x, y, *_), *arg in zip(obs['vehicles_set'], *[arg[1:] for arg in args]):
            if present < .5:
                continue

            global_pos = trans @ np.array([x * vn, y * vn, 1.])

            dx, dy = ctx.user_to_device(*global_pos[:2])

            draw_fn(ctx, (dx + 20., dy + 20.), *arg, arrow_pt=(dx, dy))

        if self.discrete:
            draw_categorical_total_expl(ctx, policy_explanation)
        else:
            draw_diag_gm_total_expl(ctx, weights, object_means, object_stds)


def draw_gaussian_gradient(ctx: cairo.Context, mu_x, mu_y, sigma_x, sigma_y, rgb, sigma_max=5., stops=11, invert=False):
    # Deliberately fail on small sigmas due to the transformation matrix becoming corrupted
    if not all((s >= .01 for s in (sigma_x, sigma_y))):
        if invert:
            ctx.set_source_rgb(*rgb)
            ctx.paint()
        return

    ctx.save()
    ctx.translate(mu_x, mu_y)
    ctx.scale(sigma_x, sigma_y)
    pattern = cairo.RadialGradient(0., 0., 0., 0., 0., sigma_max)
    for frac in np.linspace(0., 1., stops):
        density = np.exp(-.5 * (frac * sigma_max) ** 2)
        pattern.add_color_stop_rgba(frac, *rgb, 1. - density if invert else density)
    ctx.set_source(pattern)

    # TODO: Remove once resolved
    try:
        ctx.paint()
    except Exception:
        # TODO Sometimes failing with memory error here, because gaussian too small?
        print(f"{mu_x = }, {mu_y = }, {sigma_x = }, {sigma_y = }, {rgb = }, {sigma_max = }, {stops = }", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    ctx.restore()


def draw_gaussian_explanation(ctx: cairo.Context, device_pos, weight, mus, sigmas):
    dx, dy = device_pos
    ctx.save()
    ctx.identity_matrix()
    ctx.translate(dx, dy)

    ctx.scale(weight * 15, weight * 15)
    ctx.rectangle(-1., -1., 2., 2.)
    ctx.clip()

    gradient = cairo.LinearGradient(0., -1., 0., 1.)
    gradient.add_color_stop_rgb(0., 0., 1., 0.)
    gradient.add_color_stop_rgb(1., 1., 0., 0.)
    ctx.set_source(gradient)
    ctx.paint()

    draw_gaussian_gradient(ctx, *mus, *sigmas, (0., 0., 0.), stops=21, invert=True)

    ctx.reset_clip()
    ctx.restore()


def draw_multi_gaussian_explanation(ctx: cairo.Context, device_pos, weight, mus, sigmas):
    dx, dy = device_pos
    ctx.save()
    ctx.identity_matrix()
    ctx.translate(dx, dy)

    ctx.scale(weight * 15, weight * 15)
    ctx.rectangle(-1., -1., 2., 2.)
    ctx.set_source_rgb(0., 0., 0.)
    ctx.fill_preserve()
    ctx.clip()

    for m, s in zip(mus, sigmas):
        draw_gaussian_gradient(ctx, *m, *s, (1., 1., 1.), stops=11, invert=False)

    ctx.reset_clip()
    ctx.restore()


def _path_squashed_gm_expl(ctx: cairo.Context, weights, mus, sigmas):
    # K
    assert weights.shape == mus.shape
    assert weights.shape == sigmas.shape
    assert len(weights.shape) == 1

    weights = weights / np.sum(weights)

    # P
    # unsquashed_xs = np.linspace(-10., 10., 1001)
    # xs = np.tanh(unsquashed_xs)
    xs = np.linspace(-.999, .999, 1001)
    unsquashed_xs = np.arctanh(xs)

    # K
    var = np.square(sigmas)

    # P x K
    component_pdf = 1 / np.sqrt(2 * np.pi * var[None, :]) * np.exp(
        -.5 * np.square(unsquashed_xs[:, None] - mus[None, :]) / var[None, :])
    # ... x P
    pdf = np.sum(weights[None, :] * component_pdf, axis=-1)
    pdf = pdf / (1 - xs ** 2 + 1e-6)

    assert np.all(np.isfinite(pdf))
    pdf = pdf / np.max(pdf)

    ctx.new_path()
    ctx.move_to(-1, 0)
    for x, y in zip(xs, pdf):
        ctx.line_to(x, y)
    ctx.line_to(1, 0)


def draw_diag_gm_explanation(ctx: cairo.Context, device_pos, weight, mus, sigmas, arrow_pt=None):
    assert mus.shape == sigmas.shape
    assert weight.shape == mus.shape[:-1] + (1,)
    assert mus.shape[-1] == 3

    def stroke_graph(stroke_rgb, fill_rgb, line_width=1.0):
        ctx.save()
        ctx.identity_matrix()

        ctx.set_source_rgb(*fill_rgb)
        path = ctx.copy_path_flat()
        ctx.close_path()
        ctx.fill()

        ctx.append_path(path)
        ctx.set_line_width(line_width)
        ctx.set_source_rgb(*stroke_rgb)
        ctx.stroke()

        ctx.restore()

    scale = np.sum(weight) * 15

    if not (scale >= 1e-2):
        return

    if arrow_pt is not None:
        ctx.save()
        ctx.identity_matrix()
        ctx.move_to(*arrow_pt)
        ctx.line_to(*device_pos)
        ctx.set_source_rgb(0., 0., 0.)
        ctx.set_line_width(1.)
        ctx.stroke()
        ctx.restore()

    dx, dy = device_pos
    ctx.save()
    ctx.identity_matrix()
    ctx.translate(dx, dy)

    ctx.scale(scale, scale)
    ctx.rectangle(-1., -1., 2., 2.)
    ctx.set_source_rgb(0., 0., 0.)
    ctx.fill_preserve()
    ctx.clip()

    a = .65
    b = .5

    # Draw steering
    ctx.save()
    ctx.translate(.0, -.45)
    ctx.scale(a, -b)
    _path_squashed_gm_expl(ctx, np.squeeze(weight, -1), mus[..., 0], sigmas[..., 0])
    stroke_graph((1., 1., 1.), (.5, .5, .5))
    ctx.restore()

    # Draw brake
    ctx.save()
    ctx.translate(-.15, .3)
    ctx.rotate(-np.pi/2)
    ctx.scale(a, -b)
    _path_squashed_gm_expl(ctx, np.squeeze(weight, -1), mus[..., 2], sigmas[..., 2])
    stroke_graph((1., 0., 0.), (0.5, 0., 0.))
    ctx.restore()

    # Draw throttle
    ctx.save()
    ctx.translate(.15, .3)
    ctx.rotate(np.pi/2)
    ctx.scale(-a, -b)
    _path_squashed_gm_expl(ctx, np.squeeze(weight, -1), mus[..., 1], sigmas[..., 1])
    stroke_graph((0., 1., 0.), (0., 0.5, 0.))
    ctx.restore()

    ctx.restore()


def draw_diag_gm_total_expl(ctx, weights, object_means, object_stds):
    # Resulting total policy explanation
    from CarEnv.Rendering.Rendering import stroke_fill
    ctx.save()
    ctx.identity_matrix()
    ctx.rectangle(0., 0., 180., 50)
    stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))
    ctx.move_to(50., 22.)
    ctx.show_text("Resulting Mixture PDF")
    ctx.move_to(50., 36.)
    ctx.show_text("π(·|s)")
    ctx.restore()
    draw_diag_gm_explanation(ctx, (25, 25), weights.reshape((-1, 1)) / np.sum(weights),
                             object_means.reshape((-1, 3)),
                             object_stds.reshape((-1, 3)))


def draw_categorical_total_expl(ctx, policy_explanation):
    from CarEnv.Rendering.Rendering import stroke_fill
    ctx.save()
    ctx.identity_matrix()
    ctx.rectangle(0., 0., 180., 50)
    stroke_fill(ctx, (0., 0., 0.), (1., 1., 1.))
    ctx.move_to(50., 22.)
    ctx.show_text("Resulting Mixture PDF")
    ctx.move_to(50., 36.)
    ctx.show_text("π(·|s)")
    ctx.restore()
    # Draw resulting action explanation
    probs = policy_explanation['probs']
    if len(np.shape(probs)) == 2:
        probs = np.squeeze(probs, 0)
    draw_discrete_explanation(ctx, (25, 25), 1., probs / np.max(probs))


class SingleContinuousExplainableRenderCallback:
    def __init__(self):
        self.policy = None

    def __call__(self, renderer, env, ctx: cairo.Context):
        if self.policy is None:
            return

        # Despite the name, this is up to date
        sensor_obs = env.last_observations['cones_set']

        # Filter present
        sensor_obs = sensor_obs[sensor_obs[..., 0] > 0]

        vn = env.sensors['cones_set'].view_normalizer

        trans = np.linalg.inv(env.ego_transform)

        policy_explanation = self.policy(env.last_observations)
        object_means = np.array(policy_explanation['object_means'])
        object_stds = np.exp(np.array(policy_explanation['object_log_stds']))

        # O x C x 1
        weights = np.array(policy_explanation['weights'])
        # Normalize by adding components, thus max is 1 for all components of an object
        weights = weights / np.max(np.sum(weights, axis=-2))

        for (present, x, y, blue, yellow), weight, mean, std in zip(sensor_obs, weights, object_means, object_stds):
            if not (np.sum(weight) >= 1e-2):
                continue

            global_pos = trans @ np.array([x * vn, y * vn, 1.])

            dx, dy = ctx.user_to_device(global_pos[0], global_pos[1])
            # s = 10
            edx = dx + yellow * 45 - blue * 45

            if mean.shape[-1] == 3:
                draw_diag_gm_explanation(ctx, (edx, dy), weight, mean, std, arrow_pt=(dx, dy))
            elif len(mean) == 1:
                draw_gaussian_explanation(ctx, (dx, dy), weight, *mean, *std)
            else:
                draw_multi_gaussian_explanation(ctx, (dx, dy), weight, mean, std)

        draw_diag_gm_total_expl(ctx, weights, object_means, object_stds)


class TrackingContinuousExplainableRenderCallback:
    def __init__(self):
        self.policy = None

    def __call__(self, renderer, env, ctx: cairo.Context):
        from .Wrappers import CarEnvTrajectoryToSetWrapper

        if self.policy is None:
            return

        last_obs = CarEnvTrajectoryToSetWrapper.set_transform(env.last_observations)
        sensor_obs = last_obs['trajectory_set']

        # Filter present
        sensor_obs = sensor_obs[sensor_obs[..., 0] > 0]

        vn = env.sensors['trajectory'].normalizer

        trans = np.linalg.inv(env.ego_transform)

        policy_explanation = self.policy(last_obs)
        object_means = np.array(policy_explanation['object_means'])
        object_stds = np.exp(np.array(policy_explanation['object_log_stds']))

        # O x C x 1
        weights = np.array(policy_explanation['weights'])
        # Normalize by adding components, thus max is 1 for all components of an object
        weights = weights / np.max(np.sum(weights, axis=-2))

        for i, ((present, index, x, y), weight, mean, std) in enumerate(zip(sensor_obs, weights, object_means, object_stds)):
            if not (np.sum(weight) >= 1e-2):
                continue

            global_pos = trans @ np.array([x * vn, y * vn, 1.])

            dx, dy = ctx.user_to_device(global_pos[0], global_pos[1])
            ex_dx = dx + 45 * (1 if i % 2 == 0 else -1)
            # s = 10

            if mean.shape[-1] == 3:
                draw_diag_gm_explanation(ctx, (ex_dx, dy), weight, mean, std, arrow_pt=(dx, dy))
            elif len(mean) == 1:
                draw_gaussian_explanation(ctx, (dx, dy), weight, *mean, *std)
            else:
                draw_multi_gaussian_explanation(ctx, (dx, dy), weight, mean, std)

        draw_diag_gm_total_expl(ctx, weights, object_means, object_stds)
