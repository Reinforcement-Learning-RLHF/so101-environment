from envs.cup_env import CupEnv
from mujoco.viewer import launch_passive
import numpy as np
import time

class ScriptedPouringPolicy:
    def __init__(self, keyframes=None):
        # Default keyframes
        self.keyframes = keyframes or [
            {"target": np.array([0.2197, 0.8423, 0.6051, -1.3397, -1.5400, 1.3000]), "steps": 30},
            {"target": np.array([0.2243, 1.1886, -0.1385, -1.0597, -1.4800, 1.4853]), "steps": 40},
            {"target": np.array([0.2264, 1.1224, -0.1192, -1.0015, -1.4799, 0.3881]), "steps": 40},
            {"target": np.array([-0.1183, 0.4729, 0.2440, -0.5996, -1.4800, 0.3477]), "steps": 40},
            {"target": np.array([-0.1391, 0.3942, 0.3538, -0.5996, 0.2000, 0.3476]), "steps": 80}
        ]


        self.reset()

    def reset(self):
        self.current_frame = 0
        self.step_in_frame = 0
        self.start_qpos = None

    def get_action(self, obs):
        if self.current_frame >= len(self.keyframes):
            return self.keyframes[-1]["target"]

        frame = self.keyframes[self.current_frame]
        target_qpos = frame["target"]
        total_steps = frame["steps"]

        if self.step_in_frame == 0:
            self.start_qpos = obs["qpos"][:6]

        alpha = (self.step_in_frame + 1) / total_steps
        action = (1 - alpha) * self.start_qpos + alpha * target_qpos

        self.step_in_frame += 1
        if self.step_in_frame >= total_steps:
            self.step_in_frame = 0
            self.current_frame += 1

        return action


# --- Execution Loop ---
if __name__ == "__main__":
    # You can turn off offscreen rendering if you're just using the passive viewer
    env = CupEnv(render_images=False) 
    policy = ScriptedPouringPolicy()

    # Initial reset before the viewer loop starts
    obs = env.reset()
    
    with launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)

            viewer.sync()
            time.sleep(0.01)
            
            # If the environment terminates, reset both the env and the policy
            if done:
                obs = env.reset()
                policy.reset()