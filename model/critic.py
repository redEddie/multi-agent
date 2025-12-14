import torch.nn as nn
from tensordict.nn import TensorDictModule


def build_critic(env, cfg, mappo: bool = True):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    critic_in = obs_dim * env.n_agents if mappo else obs_dim
    backbone = nn.Sequential(
        nn.Linear(critic_in, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 1),
    ).to(cfg.device)

    if mappo:
        class GlobalCritic(nn.Module):
            def __init__(self, net, n_agents):
                super().__init__()
                self.net = net
                self.n_agents = n_agents

            def forward(self, obs):
                flat = obs.reshape(*obs.shape[:-2], -1)
                val = self.net(flat)
                return val.unsqueeze(-2).expand(*obs.shape[:-2], self.n_agents, 1)

        critic_module = TensorDictModule(
            module=GlobalCritic(backbone, env.n_agents),
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
        critic_module.value_module = critic_module.module
    else:
        critic_module = TensorDictModule(
            module=backbone,
            in_keys=[("agents", "observation")],
            out_keys=[("agents", "state_value")],
        )
        critic_module.value_module = backbone
    return critic_module


__all__ = ["build_critic"]
