# Command used to run this script: sudo PYTHONPATH=. $(which python) scripts/teleop.py

import numpy as np
import mujoco
import mujoco.viewer
import keyboard
from envs.arm_env import ArmEnv

def main():
    env = ArmEnv(render_images=False)
    env.reset()

    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')

    # Actuator limits from so101.xml
    lower_limits = np.array([-1.91986, -1.74533, -1.69, -1.65806, -2.74385, -0.17453])
    upper_limits = np.array([ 1.91986,  1.74533,  1.69,  1.65806,  2.84121,  1.74533])

    current_action = np.copy(env.data.qpos[:6])

    # Tuning parameters
    ik_step_size = 0.015       
    ik_rot_step_size = 0.05    
    gripper_step = 0.15   
    damping = 0.05             
    max_lead = 0.15            

    print("=== SO101 Robotic Arm Teleop (Decoupled) ===")
    print(" [POSITION]")
    print(" W/S   : Move Forward/Backward (X)")
    print(" A/D   : Move Left/Right (Y)")
    print(" Q/E   : Move Up/Down (Z)")
    print("\n [ORIENTATION]")
    print(" U/O   : Roll (Rotate around X)")
    print(" I/K   : Pitch (Rotate around Y)")
    print(" J/L   : Yaw (Rotate around Z)")
    print("\n [GRIPPER]")
    print(" SPACE : Close Gripper")
    print(" C     : Open Gripper")
    print("\n ESC   : Quit")
    print("============================================")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            if keyboard.is_pressed('esc'):
                break

            dx = np.zeros(3) 
            dr = np.zeros(3) 
            
            # Position Controls
            if keyboard.is_pressed('w'): dx[0] += ik_step_size
            if keyboard.is_pressed('s'): dx[0] -= ik_step_size
            if keyboard.is_pressed('a'): dx[1] += ik_step_size
            if keyboard.is_pressed('d'): dx[1] -= ik_step_size
            if keyboard.is_pressed('q'): dx[2] += ik_step_size
            if keyboard.is_pressed('e'): dx[2] -= ik_step_size

            # Orientation Controls
            if keyboard.is_pressed('u'): dr[0] += ik_rot_step_size
            if keyboard.is_pressed('o'): dr[0] -= ik_rot_step_size
            if keyboard.is_pressed('i'): dr[1] += ik_rot_step_size
            if keyboard.is_pressed('k'): dr[1] -= ik_rot_step_size
            if keyboard.is_pressed('j'): dr[2] += ik_rot_step_size
            if keyboard.is_pressed('l'): dr[2] -= ik_rot_step_size

            # Gripper Controls
            if keyboard.is_pressed('space'): current_action[5] += gripper_step 
            if keyboard.is_pressed('c'):     current_action[5] -= gripper_step 

            norm_dx = np.linalg.norm(dx)
            norm_dr = np.linalg.norm(dr)

            # Apply Inverse Kinematics dynamically
            if norm_dx > 0 or norm_dr > 0:
                jacp = np.zeros((3, env.model.nv))
                jacr = np.zeros((3, env.model.nv))
                mujoco.mj_jacSite(env.model, env.data, jacp, jacr, site_id)
                
                J_pos = jacp[:, :5]
                J_rot = jacr[:, :5]
                lambda_sq = damping ** 2

                if norm_dr == 0:
                    # Scenario 1: Pure Translation. 
                    # Drop rotational constraints so the 5-DOF arm doesn't fight itself.
                    J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * np.eye(3))
                    delta_q = J_dls @ dx
                else:
                    # Scenario 2: Rotation Commanded.
                    # Use the 6D constraint so it tries to hold its X/Y/Z position while rotating.
                    # (Note: Minor position drift is physically unavoidable here due to 5-DOF limitation).
                    J_full = np.vstack([J_pos, J_rot])
                    twist = np.concatenate([dx, dr])
                    J_dls = np.linalg.inv(J_full.T @ J_full + lambda_sq * np.eye(5)) @ J_full.T
                    delta_q = J_dls @ twist
                
                new_target = current_action[:5] + delta_q
                
                # ANTI-WINDUP
                actual_q = env.data.qpos[:5]
                new_target = np.clip(new_target, actual_q - max_lead, actual_q + max_lead)
                
                current_action[:5] = new_target

            # Enforce hard joint limits
            current_action = np.clip(current_action, lower_limits, upper_limits)

            # Step physics and sync viewer
            env.step(current_action)
            viewer.sync()

    env.close()

if __name__ == "__main__":
    main()