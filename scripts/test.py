import numpy as np
import torch
import os
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from envs.arm_env import ArmEnv # Ensure this is your SO101ActEnv
from random_scripted import RandomizedIKPolicy

# --- Configuration ---
REPO_ID = "Ishah8840/so101_pouring"
LOCAL_DIR = Path("data/lerobot/so101_pouring")
FPS = 50 
TOTAL_SUCCESSES_NEEDED = 200
TASK_STR = "Pour the water from the source cup into the target cup."

def collect_data():
    # Initialize Environment
    env = ArmEnv(max_steps=400)
    policy = RandomizedIKPolicy(env)

    # 1. Initialize LeRobot Dataset
    # If the dataset already exists locally, this might throw an error. 
    # Use 'create' for new datasets or 'LeRobotDataset(repo_id, root=...)' to append.
    if LOCAL_DIR.exists():
        import shutil
        print(f"Cleaning up existing data at {LOCAL_DIR}")
        shutil.rmtree(LOCAL_DIR)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        root=LOCAL_DIR,
        fps=FPS,
        robot_type="so101",
        features={
            "observation.images.front": {"dtype": "image", "shape": (3, 240, 320), "names": ["c", "h", "w"]},
            "observation.images.wrist": {"dtype": "image", "shape": (3, 240, 320), "names": ["c", "h", "w"]},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
            "action": {"dtype": "float32", "shape": (6,), "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]},
        }
    )

    success_count = 0
    attempt_count = 0

    while success_count < TOTAL_SUCCESSES_NEEDED:
        attempt_count += 1
        obs = env.reset() # Handled the (obs, info) unpack
        policy.reset()
        
        episode_buffer = []
        is_actually_successful = False

        # Run the episode
        for _ in range(400):
            # A. Get action based on the observation BEFORE stepping
            action = policy.get_action(obs)
            
            # B. Package the frame
            # Note: We transpose images to (C, H, W) for LeRobot/PyTorch
            frame = {
                "observation.images.front": obs["images/front"].transpose(2, 0, 1).astype(np.uint8),
                "observation.images.wrist": obs["images/wrist"].transpose(2, 0, 1).astype(np.uint8),
                "observation.state": np.asarray(obs["qpos"], dtype=np.float32).reshape(6),
                "action": np.asarray(action, dtype=np.float32).reshape(6),
                "task": TASK_STR,
            }
            episode_buffer.append(frame)

            # C. Step the environment
            obs, reward, done, info = env.step(action)
            
            # Check success flag from your ArmEnv logic
            if info.get("is_success", False):
                is_actually_successful = True

            if done:
                break

        # 2. Only commit to Parquet if the 'is_success' flag was ever triggered
        if is_actually_successful:
            for frame in episode_buffer:
                dataset.add_frame(frame)
            
            # save_episode() creates the parquet chunk for this trial
            dataset.save_episode()
            success_count += 1
            print(f"✅ Success {success_count}/{TOTAL_SUCCESSES_NEEDED} (Attempt {attempt_count})")
        else:
            print(f"❌ Attempt {attempt_count} failed. Discarding buffer.")

    # 3. Finalize: Calculates stats (min/max/mean/std) and writes meta.json
    print("Finalizing dataset and calculating statistics...")
    print(f"Dataset complete! Saved to {LOCAL_DIR}")

if __name__ == "__main__":
    collect_data()