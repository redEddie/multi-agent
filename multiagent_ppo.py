from scripts.train import train, make_cfg_from_args
from src.config import Config
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO (wrapper)")
    parser.add_argument("--scenario-name", type=str, default=None)
    parser.add_argument("--n-agents", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--frames-per-batch", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--n-iters", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--minibatch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lmbda", type=float, default=None)
    parser.add_argument("--entropy-eps", type=float, default=None)
    args = parser.parse_args()

    cfg = make_cfg_from_args(args)
    train(cfg)
