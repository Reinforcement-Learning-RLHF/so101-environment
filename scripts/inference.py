import numpy as np
import torch
import mujoco
import mujoco.viewer
from lerobot.policies.act.modeling_act import ACTPolicy
from envs.arm_env import ArmEnv

# --- Config ---
POLICY_PATH = "policies/20k_policy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load policy
policy = ACTPolicy.from_pretrained(POLICY_PATH)
policy.to(DEVICE)
policy.eval()

env = ArmEnv(max_steps=650)
obs = env.reset()
policy.reset()


with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running() and step < 650:
        batch = {
            "observation.state": torch.tensor(
    env.data.qpos[:6].copy(), dtype=torch.float32
).unsqueeze(0).to(DEVICE),
            "observation.images.front": torch.tensor(
                obs["images/front"], dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0,
            "observation.images.wrist": torch.tensor(
                obs["images/wrist"], dtype=torch.float32
            ).permute(2, 0, 1).unsqueeze(0).to(DEVICE) / 255.0,
        }


        with torch.no_grad():
            action = policy.select_action(batch)

        obs, reward, done, info = env.step(action.squeeze().cpu().numpy())
        
        viewer.sync()
        step += 1

        if done:
            print(f"Done — reward: {reward:.3f}, success: {info['is_success']}")
            obs = env.reset()
            policy.reset()
            step = 0