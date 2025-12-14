import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch


def save_artifacts(cfg, env, policy, critic, episode_reward_mean_list):
    """Save config, rewards, and ONNX exports under output/{scenario}/{timestamp}."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path("output") / cfg.scenario_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save rewards and config
    (out_dir / "episode_reward_mean.json").write_text(
        json.dumps(episode_reward_mean_list, indent=2)
    )
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    n_agents = env.n_agents

    def _export_onnx(module, dummy_input, path, input_name, output_name):
        module_cpu = module.to("cpu").eval()
        torch.onnx.export(
            module_cpu,
            dummy_input,
            path,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
            opset_version=18,
        )

    actor_dummy = torch.zeros(1, n_agents, obs_dim, device="cpu")
    _export_onnx(
        policy.backbone,
        actor_dummy,
        out_dir / "policy.onnx",
        input_name="observation",
        output_name="loc_scale",
    )

    critic_dummy = torch.zeros(1, n_agents, obs_dim, device="cpu")
    _export_onnx(
        critic.value_module,
        critic_dummy,
        out_dir / "critic.onnx",
        input_name="observation",
        output_name="state_value",
    )
    print(f"Artifacts saved to {out_dir}")


__all__ = ["save_artifacts"]
