import mujoco
import pkg_resources
import numpy as np
from pathlib import Path

class ArmEnv:
    def __init__(self):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.action_dim = self.model.nu      # number of actuators
        self.obs_dim = self.model.nq         # number of joint positions

        self.action_space = np.zeros(self.action_dim)
        self.observation_space = np.zeros(self.obs_dim)

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        target_id = self.model.body("target_cup").id
        source_id = self.model.body("source_cup").id

        target_x = 0.4 + np.random.uniform(-0.05, 0.05)
        target_y = 0.20 + np.random.uniform(-0.05, 0.05)
        self.model.body_pos[target_id][:2] = [target_x, target_y]

        source_x = 0.4 + np.random.uniform(-0.05, 0.05)
        source_y = -0.20 + np.random.uniform(-0.05, 0.05)
        
        cup_qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        self.data.qpos[cup_qpos_adr : cup_qpos_adr + 2] = [source_x, source_y]
       
        dx = source_x - 0.4
        dy = source_y - (-0.2)

        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "water_particle" in name:
                jnt_adr = self.model.jnt_qposadr[self.model.body_jntadr[i]]
                self.data.qpos[jnt_adr] += dx
                self.data.qpos[jnt_adr + 1] += dy

        table_id = self.model.geom("table_top").id
        self.model.geom_friction[table_id][0] = np.random.uniform(0.8, 1.2)

        self.model.body_mass[source_id] = np.random.uniform(0.04, 0.07)

        for i in range(self.model.ngeom):
            if self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.model.geom_friction[i][0] = np.random.uniform(0.005, 0.02)

        self.model.light_diffuse[0] = np.random.uniform(0.5, 0.8, 3)

        cam_id = self.model.camera("main_observation").id
        base_pos = np.array([1.2, 0.0, 1.3])
        self.model.cam_pos[cam_id] = base_pos + np.random.uniform(-0.02, 0.02, 3)

        mujoco.mj_forward(self.model, self.data)

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