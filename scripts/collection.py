import numpy as np
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from envs.arm_env import ArmEnv
from random_scripted import RandomizedIKPolicy

REPO_ID = "Ishah8840/so101_pouring"
LOCAL_DIR = Path("data/lerobot/so101_pouring")
FPS = 50
TOTAL_SUCCESSES_NEEDED = 60
TASK_STR = "Pour the water from the source cup into the target cup."

def collect_data():
    env = ArmEnv(max_steps=220)
    policy = RandomizedIKPolicy(env)

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
            "observation.images.front": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.images.wrist": {
                "dtype": "image",
                "shape": (240, 320, 3),
                "names": ["height", "width", "channel"]
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            },
            "action": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
            },
        }
    )

    success_count = 0
    attempt_count = 0

    while success_count < TOTAL_SUCCESSES_NEEDED:
        attempt_count += 1

        obs, info = env.reset()  # ✅ unpack (obs, info)
        policy.reset()

        episode_buffer = []
        is_actually_successful = False

        for _ in range(220):
            action = policy.get_action(obs)

            frame = {
                "observation.images.front": obs["pixels"]["front"].astype(np.uint8),   # ✅ new key
                "observation.images.wrist": obs["pixels"]["wrist"].astype(np.uint8),   # ✅ new key
                "observation.state": obs["agent_pos"].copy().astype(np.float32),        # ✅ from obs, raw radians
                "action": action.astype(np.float32),                                    # ✅ raw radians, no denorm
                "task": TASK_STR,
            }

            episode_buffer.append(frame)

            obs, reward, terminated, truncated, info = env.step(action)  # ✅ 5 values

            if info.get("is_success", False):
                is_actually_successful = True

        if is_actually_successful:
            for frame in episode_buffer:
                dataset.add_frame(frame)
            dataset.save_episode()
            success_count += 1
            print(f"✅ Episode {success_count} saved — {len(episode_buffer)} frames")
        else:
            print(f"❌ Attempt {attempt_count} failed. Discarding.")

    print("Finalizing dataset...")
    dataset.finalize()
    print(f"Done! Saved to {LOCAL_DIR}")

if __name__ == "__main__":
    collect_data()