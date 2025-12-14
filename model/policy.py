import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch.distributions import constraints
import torchrl.modules


class TanhNormalBound(torch.distributions.transformed_distribution.TransformedDistribution):
    """TorchRL 0.10 passes min/max; this subclass ignores them and sums log-probs."""

    def __init__(self, loc, scale, min=None, max=None, validate_args=None):
        base_dist = torch.distributions.Normal(loc, scale)
        transforms = [torch.distributions.transforms.TanhTransform()]
        super().__init__(base_dist, transforms, validate_args=validate_args)
        # cache for RSample
        self.loc = loc
        self.scale = scale
        self._min = min
        self._max = max

    @property
    def mean(self):
        return torch.tanh(self.loc)

    @property
    def support(self):
        return constraints.interval(-1.0, 1.0)

    @property
    def mode(self):
        # For a symmetric normal, mode == mean
        return self.mean

    def log_prob(self, value):
        log_prob = super().log_prob(value)
        # Aggregate per-dimension log-probabilities for multi-dimensional actions
        if log_prob.shape != self.loc.shape:
            return log_prob
        return log_prob.sum(dim=-1)


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


__all__ = ["build_policy", "TanhNormalBound"]
