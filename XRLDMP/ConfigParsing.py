import omegaconf


def parse_sac(cfg: omegaconf.DictConfig):
    kwargs = dict(gamma=cfg.agent.gamma, tau=cfg.agent.tau, target_entropy=cfg.agent.target_entropy,
                  lr=cfg.agent.get('lr', 3e-4), q_lr=cfg.agent.get('q_lr', None),
                  ent_coef=cfg.agent.alpha, net_arch=(cfg.agent.net_width,) * cfg.agent.net_depth,
                  critic_max_grad_norm=cfg.agent.clip_norm)
    return kwargs


def parse_xdsac(cfg: omegaconf.DictConfig, batch_obs):
    kwargs = dict(collate_fn=batch_obs, gamma=cfg.agent.gamma, tau=cfg.agent.tau,
                  uniform_mixture=cfg.agent.uniform_mixture, lr=cfg.agent.get('lr', 3e-4),
                  q_lr=cfg.agent.get('q_lr', None),
                  alpha="auto" if cfg.agent.alpha is None else cfg.agent.alpha,
                  target_entropy=cfg.agent.target_entropy,
                  actor_net_arch=(cfg.agent.actor_width,) * cfg.agent.actor_depth,
                  critic_net_arch=(cfg.agent.critic_width,) * cfg.agent.critic_depth,
                  kappa=cfg.agent.kappa, clip_norm=cfg.agent.clip_norm)
    return kwargs


def parse_xsac(cfg: omegaconf.DictConfig):
    kwargs = dict(gamma=cfg.agent.gamma, tau=cfg.agent.tau, lr=cfg.agent.lr, q_lr=cfg.agent.get('q_lr', None),
                  uniform_mixture=cfg.agent.uniform_mixture,
                  critic_net_arch=(cfg.agent.critic_width,) * cfg.agent.critic_depth,
                  ent_coef=cfg.agent.alpha, actor_net_arch=(cfg.agent.actor_width,) * cfg.agent.actor_depth,
                  kappa=cfg.agent.kappa, clip_norm=cfg.agent.clip_norm, target_entropy=cfg.agent.target_entropy,
                  components_per_object=cfg.agent.components_per_object,
                  weight_latent_depth=cfg.agent.weight_latent_depth,)
    return kwargs


def parse_dsac(cfg: omegaconf.DictConfig, batch_obs):
    assert cfg.agent.type == "dsac"

    kwargs = dict(collate_fn=batch_obs, gamma=cfg.agent.gamma, tau=cfg.agent.tau,
                  alpha="auto" if cfg.agent.alpha is None else cfg.agent.alpha,
                  lr=cfg.agent.lr,
                  target_entropy=cfg.agent.target_entropy, duelling=cfg.agent.duelling,
                  clip_norm=cfg.agent.clip_norm, net_arch=(cfg.agent.net_width,) * cfg.agent.net_depth)
    return kwargs
