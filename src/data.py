import torch
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter


def build_collector(env, policy, cfg):
    return SyncDataCollector(
        env,
        policy,
        device=torch.device(cfg.device),
        policy_device=torch.device(cfg.device),
        storing_device=torch.device(cfg.device),
        frames_per_batch=cfg.frames_per_batch,
        total_frames=cfg.frames_per_batch * cfg.n_iters,
    )


def build_replay_buffer(cfg):
    return ReplayBuffer(
        storage=ListStorage(cfg.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
        batch_size=cfg.minibatch_size,
        writer=RoundRobinWriter(compilable=True),
    )


__all__ = ["build_collector", "build_replay_buffer"]
