import argparse
import torch
import torch.nn as nn
from tqdm import tqdm

from model.critic import build_critic
from model.policy import build_policy
from src.config import Config
from src.data import build_collector, build_replay_buffer
from src.env import make_env
from src.losses import build_loss
from util.save import save_artifacts


def make_cfg_from_args(args):
    cfg = Config()
    for field in cfg.__dataclass_fields__:
        if hasattr(args, field) and getattr(args, field) is not None:
            setattr(cfg, field, getattr(args, field))
    return cfg


def train(cfg: Config):
    if cfg.device.startswith("cuda"):
        try:
            avail = torch.cuda.is_available()
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0) if count > 0 else "N/A"
            print(f"CUDA available: {avail}, device_count: {count}, device0: {name}")
        except Exception as e:
            raise RuntimeError(f"CUDA probe failed: {e}") from e
        if not avail:
            raise RuntimeError("CUDA is not available but GPU is enforced.")

    env = make_env(cfg)
    policy = build_policy(env, cfg)
    critic = build_critic(env, cfg, mappo=True)
    collector = build_collector(env, policy, cfg)
    replay_buffer = build_replay_buffer(cfg)
    loss_module = build_loss(env, policy, critic, cfg)
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.lr)

    pbar = tqdm(total=cfg.n_iters, desc="episode_reward_mean = 0")
    episode_reward_mean_list = []

    for _ in range(cfg.n_iters):
        for tensordict_data in collector:
            # Align done/terminated shapes with per-agent rewards for GAE
            reward = tensordict_data.get(("next", *env.reward_key))
            for key in [("next", "done"), ("next", "terminated")]:
                if key in tensordict_data.keys(include_nested=True):
                    done = tensordict_data.get(key)
                    if done.shape != reward.shape:
                        expanded = done.unsqueeze(-2)
                        expanded = expanded.expand(
                            *expanded.shape[:-2], reward.shape[-2], expanded.shape[-1]
                        )
                        tensordict_data.set(key, expanded)
            replay_buffer.extend(tensordict_data)

            # Compute advantage and value targets
            for _ in range(cfg.num_epochs):
                for batch in replay_buffer:
                    loss_module.value_estimator(batch)

                    loss_vals = loss_module(batch)
                    loss_actor = loss_vals["loss_objective"]
                    loss_critic = loss_vals["loss_critic"]
                    loss_entropy = loss_vals["loss_entropy"]
                    loss = loss_actor + loss_critic + loss_entropy

                    optim.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(loss_module.parameters(), cfg.max_grad_norm)
                    optim.step()
            replay_buffer.empty()
            break  # one batch per iter

        # Eval on last batch reward; fallback to raw reward if episode_reward is absent
        ep_rew = tensordict_data.get(("next", "agents", "episode_reward"), None)
        if ep_rew is None:
            ep_rew = tensordict_data.get(("next", *env.reward_key)).mean()
        else:
            ep_rew = ep_rew.mean()
        episode_reward_mean_list.append(ep_rew.item())
        pbar.set_description(f"episode_reward_mean = {ep_rew.item():.3f}")
        pbar.update(1)

    pbar.close()
    print("Training done. Mean episode reward history:", episode_reward_mean_list)

    save_artifacts(cfg, env, policy, critic, episode_reward_mean_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on VMAS scenarios")
    parser.add_argument("--scenario-name", type=str, default=None, help="VMAS scenario (e.g., navigation, wheel)")
    parser.add_argument("--n-agents", type=int, default=None, help="Number of agents")
    parser.add_argument("--device", type=str, default=None, help="Device string, e.g., cpu or cuda:0")
    parser.add_argument("--frames-per-batch", type=int, default=None, help="Frames per batch for collection")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode")
    parser.add_argument("--n-iters", type=int, default=None, help="Number of iterations (batches)")
    parser.add_argument("--num-epochs", type=int, default=None, help="PPO epochs per batch")
    parser.add_argument("--minibatch-size", type=int, default=None, help="PPO minibatch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=None, help="Max grad norm")
    parser.add_argument("--clip-epsilon", type=float, default=None, help="PPO clip epsilon")
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=None, help="GAE lambda")
    parser.add_argument("--entropy-eps", type=float, default=None, help="Entropy coefficient")
    args = parser.parse_args()

    cfg = make_cfg_from_args(args)
    train(cfg)
