import argparse
import json
from pathlib import Path

import onnxruntime as ort
import torch
from tensordict import TensorDict

from model.policy import build_policy
from src.config import Config
from src.env import make_env


def load_latest_run(scenario: str):
    base = Path("output") / scenario
    if not base.exists():
        raise FileNotFoundError(f"No output directory found for scenario {scenario}")
    runs = sorted(base.iterdir())
    if not runs:
        raise FileNotFoundError(f"No runs found under {base}")
    return runs[-1]


def _load_onnx_session(run_dir: Path):
    onnx_path = run_dir / "policy.onnx"
    if not onnx_path.exists():
        return None
    session = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    outputs = session.get_outputs()
    loc_name = None
    scale_name = None
    for out in outputs:
        name = out.name.lower()
        if "loc" in name:
            loc_name = out.name
        if "scale" in name or "clamp" in name:
            scale_name = out.name
    if loc_name is None and outputs:
        loc_name = outputs[0].name
    if scale_name is None and len(outputs) > 1:
        scale_name = outputs[1].name
    return session, input_name, loc_name, scale_name


def play(
    run_dir: Path,
    device: str,
    n_steps: int = 200,
    render: bool = False,
    num_envs: int = 1,
):
    # Load config
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {run_dir}")
    cfg_dict = json.loads(cfg_path.read_text())
    cfg = Config(**cfg_dict)
    cfg.device = device  # override if needed
    # For playback, use a small number of envs to match ONNX export shape
    cfg.frames_per_batch = cfg.max_steps * num_envs

    env = make_env(cfg)
    env.to(device)

    session_info = _load_onnx_session(run_dir)
    policy = None
    if session_info is None:
        # Fallback to torch checkpoint
        policy = build_policy(env, cfg)
        pt_path = run_dir / "policy.pt"
        if not pt_path.exists():
            raise FileNotFoundError(
                f"policy.onnx not found (preferred) and policy.pt missing in {run_dir}; cannot play back"
            )
        state_dict = torch.load(pt_path, map_location=device)
        policy.load_state_dict(state_dict)
        policy.to(device)

    td = env.reset()
    rewards = []
    ep_reward_sum = None
    if session_info is not None:
        session, input_name, loc_name, scale_name = session_info
    else:
        session = None
    frames = []  # always record for gif
    for _ in range(n_steps):
        if session is not None:
            obs = td.get(("agents", "observation"))
            obs_np = obs.detach().cpu().numpy()
            ort_outs = session.run(
                [name for name in [loc_name, scale_name] if name is not None],
                {input_name: obs_np},
            )
            loc = torch.from_numpy(ort_outs[0]).to(device)
            action = torch.tanh(loc)  # deterministic action from mean
            td.set(env.action_key, action)
        else:
            with torch.no_grad():
                td = policy(td)
        td = env.step(td)
        maybe_ep = td.get(("next", "agents", "episode_reward"), None)
        if maybe_ep is not None:
            ep_reward_sum = maybe_ep.mean().item()
        rewards.append(td.get(("next", *env.reward_key)).mean().item())
        td = td.get("next")
        if render:
            try:
                env.render()
            except Exception:
                pass
        try:
            frame = env.render(mode="rgb_array")
        except Exception:
            frame = None
        if frame is not None:
            frames.append(frame)
    step_mean = sum(rewards) / len(rewards) if rewards else 0.0
    step_sum = sum(rewards)
    msg = f"Step reward mean: {step_mean:.4f} | step reward sum: {step_sum:.4f}"
    if ep_reward_sum is not None:
        msg += f" | episode reward sum: {ep_reward_sum:.4f}"
    print(msg)
    if frames:
        import imageio

        gif_path = run_dir / "rollout.gif"
        imageio.mimsave(gif_path, frames, fps=15)
        print(f"Saved rollout to {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained policy from the latest run directory")
    parser.add_argument("--scenario-name", type=str, required=True, help="Scenario name (e.g., navigation)")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific run directory under output/<scenario>")
    parser.add_argument("--device", type=str, default="cpu", help="Device for rollout (e.g., cpu or cuda:0)")
    parser.add_argument("--n-steps", type=int, default=200, help="Number of steps to roll out")
    parser.add_argument("--render", action="store_true", help="Render if supported by env (may fail headless)")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs for playback (default 1)")
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = load_latest_run(args.scenario_name)

    play(
        run_dir,
        device=args.device,
        n_steps=args.n_steps,
        render=args.render,
        num_envs=args.num_envs,
    )
