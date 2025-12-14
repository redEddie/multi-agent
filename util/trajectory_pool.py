import contextlib
import multiprocessing as mp

import torch
import torchrl.collectors.collectors as collectors


class NoShmTrajectoryPool:
    """Trajectory pool that avoids shared memory usage."""

    def __init__(self, ctx=None, lock: bool = False):
        self.ctx = ctx
        self._traj_id = torch.zeros((), device="cpu", dtype=torch.int)
        if ctx is None:
            self.lock = contextlib.nullcontext() if not lock else mp.RLock()
        else:
            self.lock = contextlib.nullcontext() if not lock else ctx.RLock()

    def get_traj_and_increment(self, n=1, device=None):
        with self.lock:
            v = self._traj_id.item()
            out = torch.arange(v, v + n).to(device)
            self._traj_id.copy_(torch.tensor(1 + out[-1].item()))
        return out


# Override the TorchRL collector trajectory pool to avoid shared memory.
collectors._TrajectoryPool = NoShmTrajectoryPool

__all__ = ["NoShmTrajectoryPool"]
