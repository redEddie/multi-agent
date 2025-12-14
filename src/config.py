from dataclasses import dataclass


@dataclass
class Config:
    # devices
    device: str = "cuda:0"  # force GPU; will error if unavailable
    # env
    max_steps: int = 100
    frames_per_batch: int = 6_000
    n_iters: int = 10
    scenario_name: str = "navigation"
    n_agents: int = 3
    # training
    num_epochs: int = 30
    minibatch_size: int = 400
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    # PPO
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    lmbda: float = 0.9
    entropy_eps: float = 1e-4
