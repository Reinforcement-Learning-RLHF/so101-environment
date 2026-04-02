import torch
import mujoco
import mujoco.viewer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from envs.arm_env import ArmEnv

POLICY_PATH = "policies/50k_policy"
DATASET_PATH = "/home/ishan-shah/Projects/preference-learning-so101/data/lerobot/so101_pouring"  # needed for normalization stats
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load policy
policy = ACTPolicy.from_pretrained(POLICY_PATH)
policy.to(DEVICE)
policy.eval()

# Load preprocessor/postprocessor using your dataset's stats
dataset = LeRobotDataset(DATASET_PATH)

preprocessor, postprocessor = make_pre_post_processors(
    policy.config,
    pretrained_path=POLICY_PATH,
    dataset_stats=dataset.meta.stats
)

env = ArmEnv(max_steps=200)
obs, info = env.reset()
policy.reset()

with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    step = 0
    while viewer.is_running() and step < 200:

        state_tensor = torch.from_numpy(obs["agent_pos"]).float().unsqueeze(0).to(DEVICE)

        front_img = torch.from_numpy(obs["pixels"]["front"]).float().permute(2, 0, 1) / 255.0
        front_img = front_img.unsqueeze(0).to(DEVICE)

        wrist_img = torch.from_numpy(obs["pixels"]["wrist"]).float().permute(2, 0, 1) / 255.0
        wrist_img = wrist_img.unsqueeze(0).to(DEVICE)

        batch = {
            "observation.state": state_tensor,
            "observation.images.front": front_img,
            "observation.images.wrist": wrist_img,
        }

        policy_input = preprocessor(batch)

        with torch.no_grad():
            action = policy.select_action(policy_input)

        # ✅ Postprocessor converts back to raw radians for your env
        raw_action = postprocessor(action).squeeze().cpu().numpy()

        obs, reward, terminated, truncated, info = env.step(raw_action)  # ✅ 5 values

        viewer.sync()
        step += 1

        if (terminated or truncated):  # ✅ check both
            print(f"Done — reward: {reward:.3f}, success: {info['is_success']}")
            obs, info = env.reset()
            policy.reset()
            step = 0