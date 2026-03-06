from envs.arm_env import ArmEnv
from mujoco.viewer import launch
import numpy as np
from mujoco.viewer import launch_passive
import time

def main():
    env = ArmEnv()
    
    # launch_passive doesn't block execution
    with launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            # 1. Trigger your randomization logic
            env.reset() 
            
            viewer.sync()
            
            # 3. Run a few steps to see it
            for _ in range(500):
                env.step(np.zeros(env.action_dim))
                viewer.sync()

if __name__ == "__main__":
    main()