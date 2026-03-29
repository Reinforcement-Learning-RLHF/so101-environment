import mujoco
import numpy as np
import collections
from pathlib import Path

class ArmEnv:
    def __init__(self, render_width=320, render_height=240, max_steps=400):
        project_root = Path(__file__).parent.parent.resolve()
        xml_path = str(project_root / 'models' / 'scene.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.action_dim = len(self.joint_names)
        self.joint_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        }
        self.n_substeps = 10
        self.max_steps = max_steps
        self.curr_step = 0

        self.render_width = render_width
        self.render_height = render_height
        self.renderer = mujoco.Renderer(self.model, height=render_height, width=render_width)

        self.particle_indices = [
            i for i in range(self.model.nbody)
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            and "water_particle" in mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
        ]

    def get_image(self, cam_name):
        self.renderer.update_scene(self.data, camera=cam_name)
        return self.renderer.render()

    def get_obs(self):
        obs = collections.OrderedDict()
        obs['qpos'] = self.data.qpos[:6].copy().astype(np.float32)
        obs['qvel'] = self.data.qvel[:6].copy().astype(np.float32)
        # source cup pose (7) — what ACT logs as env_state for conditioning
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        obs['env_state'] = self.data.qpos[s_q_addr : s_q_addr + 7].copy().astype(np.float32)
        obs['images'] = {
            'main': self.get_image("main_observation"),   # (H, W, 3) uint8
            'wrist': self.get_image("wrist_cam"),
        }
        return obs

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.joint_ids["wrist_roll"]] = -np.pi/2

        # Randomize target cup
        target_id = self.model.body("target_cup").id
        self.model.body_pos[target_id][:2] = [
            0.5 + np.random.uniform(-0.02, 0.02),
            0.1 + np.random.uniform(-0.02, 0.02)
        ]

        # Randomize source cup
        source_id = self.model.body("source_cup").id
        s_q_addr = self.model.jnt_qposadr[self.model.body_jntadr[source_id]]
        s_x = 0.5 + np.random.uniform(-0.02, 0.02)
        s_y = -0.1 + np.random.uniform(-0.02, 0.02)
        self.data.qpos[s_q_addr : s_q_addr + 7] = [s_x, s_y, 0.422, 1, 0, 0, 0]

        # Place particles in 3D grid above source cup
        particle_count = 0
        grid_side = 3
        spacing = 0.009
        for i in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "water_particle" in name:
                jnt_addr = self.model.jnt_qposadr[self.model.body_jntadr[i]]
                ix = (particle_count % grid_side) - 1
                iy = ((particle_count // grid_side) % grid_side) - 1
                iz = (particle_count // (grid_side * grid_side))
                self.data.qpos[jnt_addr : jnt_addr + 3] = [
                    s_x + ix * spacing,
                    s_y + iy * spacing,
                    0.435 + iz * spacing
                ]
                v_addr = self.model.jnt_dofadr[self.model.body_jntadr[i]]
                self.data.qvel[v_addr : v_addr + 6] = 0.0
                particle_count += 1

        mujoco.mj_forward(self.model, self.data)
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)

        self.curr_step = 0
        return self.get_obs()

    def step(self, action):
        self.curr_step += 1

        noise = np.random.normal(0, 0.0002, size=action.shape)
        ctrl = np.clip(action + noise, -3.14, 3.14)
        self.data.ctrl[:] = ctrl

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()

        target_pos = self.data.xpos[self.model.body("target_cup").id]
        cup_halfsize = np.array([0.02, 0.02, 0.033])
        p_pos = self.data.xpos[self.particle_indices]
        inside = np.all(np.abs(p_pos - target_pos) <= cup_halfsize, axis=1)
        success_ratio = np.mean(inside)

        reward = float(success_ratio)
        done = bool(success_ratio > 0.9) or bool(self.curr_step >= self.max_steps)
        info = {"is_success": bool(success_ratio > 0.9)}

        return obs, reward, done, info

    def close(self):
        self.renderer.close()