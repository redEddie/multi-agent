import torch
from torchrl.envs import TransformedEnv, RewardSum
from torchrl.envs.libs.vmas import VmasEnv

from util.trajectory_pool import NoShmTrajectoryPool  # noqa: F401


def make_env(cfg):
    num_vmas_envs = cfg.frames_per_batch // cfg.max_steps
    base_env = VmasEnv(
        scenario=cfg.scenario_name,
        num_envs=num_vmas_envs,
        continuous_actions=True,
        max_steps=cfg.max_steps,
        device=torch.device(cfg.device),
        n_agents=cfg.n_agents,
    )
    # Accumulate per-episode reward for logging
    env = TransformedEnv(
        base_env,
        RewardSum(
            in_keys=[base_env.reward_key],
            out_keys=[("agents", "episode_reward")],
        ),
    )
    return env


__all__ = ["make_env"]
