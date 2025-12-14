import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torchrl.modules.distributions import TanhNormal
import torchrl.modules

class TanhNormalBound(TanhNormal):
      def __init__(self, *args, min=None, max=None, **kwargs):
          super().__init__(*args, **kwargs)

def build_policy(env, cfg):
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    act_dim = env.unbatched_action_spec[env.action_key].shape[-1]
    net = nn.Sequential(
        nn.Linear(obs_dim, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 2 * act_dim),
        NormalParamExtractor(),
    )
    module = TensorDictModule(
        net.to(cfg.device),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    policy = torchrl.modules.ProbabilisticActor(
        module=module,
        spec=env.unbatched_action_spec,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormalBound,
        return_log_prob=True,
        log_prob_key=("agents", "sample_log_prob"),
    )
    policy.backbone = net
    return policy


__all__ = ["build_policy"]
