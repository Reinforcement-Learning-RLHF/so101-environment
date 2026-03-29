import torch
import numpy as np
import torch.nn.functional as F
import mujoco.viewer
from lerobot.policies.act.modeling_act import ACTPolicy
from envs.arm_env import ArmEnv 

def run_test():
    ckpt_path = "so101_100k_policy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = ACTPolicy.from_pretrained(ckpt_path)
    policy.eval()
    policy.to(device)

    env = ArmEnv(render_images=True)
    obs = env.reset()

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            policy_input = {
                "observation.images.main": torch.from_numpy(obs["image_main"]).unsqueeze(0).to(device).float(),
                "observation.images.wrist": torch.from_numpy(obs["image_wrist"]).unsqueeze(0).to(device).float(),
                "observation.state": torch.from_numpy(obs["qpos"]).unsqueeze(0).to(device).float(),
            }

            with torch.no_grad():
                action = policy.select_action(policy_input).cpu().numpy()

            obs, reward, done, info = env.step(action)
            viewer.sync()

            if done:
                print(f"Done — success: {info['is_success']}")
                obs = env.reset()

if __name__ == "__main__":
    run_test()