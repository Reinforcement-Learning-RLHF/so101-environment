import mujoco
import numpy as np
from pathlib import Path
from gymnasium import spaces

class ArmEnv:
    def __init__(self, render_width=640, render_height=480, max_steps=400):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        # Dimensions
        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.action_dim = len(self.joint_names)
        self.obs_dim = len(self.joint_names)

        self.n_substeps = 10

        # Renderer
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)
        self.cam_names = ["main_observation", "wrist_cam"]

        # Gym-style spaces
        self.action_space = spaces.Box(low=-3.14, high=3.14, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Episode tracking
        self.max_steps = max_steps
        self.curr_step = 0
        self.episode_id = None

    def get_image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        img = self.renderer.render()
        # Normalize to [0,1] and transpose to (C, H, W) for ACT/PyTorch
        img = np.transpose(img, (2, 0, 1)) / 255.0  
        return img.astype(np.float32)

    def get_obs(self):
        qpos = np.array([self.data.joint(name).qpos[0] for name in self.joint_names], dtype=np.float32)
        qvel = np.array([self.data.joint(name).qvel[0] for name in self.joint_names], dtype=np.float32)

        main_image = self.get_image("main_observation")
        wrist_image = self.get_image("wrist_cam")
        
        return {
            "qpos": qpos,
            "qvel": qvel,
            "image_main": main_image,
            "image_wrist": wrist_image,
            "episode_id": self.episode_id,
            "step": self.curr_step
        }

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.episode_id = np.random.randint(0, int(1e6))

        # 1. Randomize target cup (Fixed body, so body_pos is fine)
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [0.4 + np.random.uniform(-0.05, 0.05), 
                                            0.2 + np.random.uniform(-0.05, 0.05)]

        # 2. Randomize source cup (Freejoint, must use qpos)
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.4 + np.random.uniform(-0.05, 0.05)
        s_y = -0.2 + np.random.uniform(-0.05, 0.05)
        # [x, y, z, qw, qx, qy, qz]
        self.data.qpos[s_q_addr : s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]

        # 3. FIXED: Move particles into a 3D grid (Prevents overlapping explosions)
        particle_count = 0
        grid_side = 3  # A 3x3 base
        spacing = 0.009 # Particles are size 0.004 (dia 0.008), so 0.009 gives 1mm gap
        
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "water_particle" in name:
                jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[i]]
                
                # Calculate grid offsets
                ix = (particle_count % grid_side) - 1
                iy = ((particle_count // grid_side) % grid_side) - 1
                iz = (particle_count // (grid_side * grid_side))
                
                self.data.qpos[jnt_addr : jnt_addr + 3] = [
                    s_x + (ix * spacing), 
                    s_y + (iy * spacing), 
                    0.435 + (iz * spacing)
                ]
                
                # Reset velocities to zero to clear previous momentum
                v_addr = self.model.jnt_dofadr[self.model.body_jntadr[i]]
                self.data.qvel[v_addr : v_addr + 6] = 0.0
                particle_count += 1

        # 4. Domain Randomization (Friction/Mass/Light)
        table_id = self.model.geom("table_top").id
        self.model.geom_friction[table_id][0] = np.random.uniform(0.8, 1.2)
        self.model.body_mass[source_id] = np.random.uniform(0.04, 0.07)
        
        # Randomize light and camera
        self.model.light_diffuse[0] = np.random.uniform(0.5, 0.8, 3)
        cam_id = self.model.camera("main_observation").id
        self.model.cam_pos[cam_id] = np.array([1.2, 0.0, 1.3]) + np.random.uniform(-0.02, 0.02, 3)

        # 5. CRITICAL: Settle Period
        # Teleporting objects creates "numerical shock." 
        # We run 50 steps of pure physics to let the water drop into the cup.
        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
            
        self.curr_step = 0 # Reset step count after settling
        return self.get_obs()

    def step(self, action):
        self.curr_step += 1

        # Apply action
        for _ in range(self.n_substeps):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()

        # Check particles inside target cup
        target_pos = self.data.xpos[self.model.body("target_cup").id]
        cup_halfsize = np.array([0.02, 0.02, 0.033])

        particle_indices = [i for i in range(self.model.nbody)
                            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
                            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)]

        p_pos = self.data.xpos[particle_indices]
        inside = np.all(np.abs(p_pos - target_pos) <= cup_halfsize, axis=1)
        
        success_ratio = np.mean(inside)
        done = success_ratio > 0.9 or self.curr_step >= self.max_steps
        
        # Reward for evaluation (ACT doesn't use it for training)
        reward = success_ratio 

        return obs, reward, done, {"is_success": success_ratio > 0.9}

    def close(self):
        pass