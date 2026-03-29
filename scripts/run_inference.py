import time
from pathlib import Path
from envs.arm_env import ArmEnv
from lerobot.policies.act.modeling_act import ACTPolicy
from mujoco.viewer import launch_passive
import numpy as np
import torch

# Path to your pretrained policy
policy_dir = Path("policies/so101_100k_policy")


def format_obs_for_act(obs, device="cpu"):
    """
    Reformat ArmEnv observation dict to the keys expected by ACT,
    converting everything to torch tensors.
    """
    act_obs = {
        # Only qpos for ACT robot state
        "observation.state": torch.tensor(
            obs["qpos"],  # <--- only positions, not qvel
            dtype=torch.float32,
            device=device
        ).unsqueeze(0),  # add batch dimension
        "observation.images.main": torch.tensor(
            obs["image_main"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "observation.images.wrist": torch.tensor(
            obs["image_wrist"], dtype=torch.float32, device=device
        ).unsqueeze(0),
        "episode_id": torch.tensor([obs["episode_id"]], dtype=torch.int64, device=device),
        "step": torch.tensor([obs["step"]], dtype=torch.int64, device=device)
    }
    return act_obs


if __name__ == "__main__":
    env = ArmEnv(render_images=True)
    policy = ACTPolicy.from_pretrained(policy_dir)
    obs = env.reset()
    policy.reset()

    print("Launching Passive Viewer... Press ESC in viewer to quit.")

    with launch_passive(env.model, env.data) as viewer:
        done = False
        while viewer.is_running():
            # Wrap observation for ACT
            act_obs = format_obs_for_act(obs, device="cpu")
            action = policy.select_action(act_obs)
            action = action.squeeze(0).numpy()

            # Step environment
            obs, reward, done, info = env.step(action)

            # Refresh viewer
            viewer.sync()

            if done:
                print("Episode finished. Resetting...")
                obs = env.reset()
                policy.reset()