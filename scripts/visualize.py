from envs.arm_env import ArmEnv
from mujoco.viewer import launch

def main():
    env = ArmEnv()
    launch(env.model, env.data)

if __name__ == "__main__":
    main()