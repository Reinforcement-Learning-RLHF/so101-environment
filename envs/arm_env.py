import mujoco
import pkg_resources
import numpy as np
from pathlib import Path

class ArmEnv:
    def __init__(self):
        # Load the XML model from your package
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.action_dim = self.model.nu      # number of actuators
        self.obs_dim = self.model.nq         # number of joint positions

        self.action_space = np.zeros(self.action_dim)
        self.observation_space = np.zeros(self.obs_dim)

    def reset(self):
        """Reset the simulation and return initial observation"""
        mujoco.mj_resetData(self.model, self.data)
        return self.get_obs()

    def step(self, action):
        """
        Apply an action and step the simulation
        Returns: obs, reward, done, info
        """
        action = np.array(action)
        self.data.ctrl[:] = action

        mujoco.mj_step(self.model, self.data)

        reward = 0.0
        done = False
        info = {}

        return self.get_obs(), reward, done, info

    def get_obs(self):
        """Return the current observation (joint positions)"""
        return self.data.qpos.copy()

    def close(self):
        """Optional cleanup"""
        pass