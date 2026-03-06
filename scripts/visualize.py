from envs.arm_env import ArmEnv
from mujoco.viewer import launch
import numpy as np
from mujoco.viewer import launch_passive
import time


def main():
    env = ArmEnv()
    
    with launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            env.reset()
            
            random_action = np.random.uniform(low=-0.5, high=0.5, size=env.action_dim)
            
            for _ in range(100):
                obs, reward, done, info = env.step(random_action) 
                viewer.sync()
                time.sleep(0.01)

if __name__ == "__main__":
    main()