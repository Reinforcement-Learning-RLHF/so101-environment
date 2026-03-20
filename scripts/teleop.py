import time
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

    target_q = np.copy(env.data.qpos[:6])

    # --- TUNING PARAMETERS ---
    control_hz = 60
    dt = 1.0 / control_hz
    
    base_ik_step = 0.015               # Translation speed
    base_pitch_step = 0.1              # Pitch speed (Tip up/down)
    base_roll_step = 0.1               # Roll speed (Twist)
    base_gripper = 0.10                 # Gripper speed
    damping = 0.05                      # DLS IK damping
    max_lead = 0.05                     # Anti-windup limit
    # -------------------------

    print("=== SO101 Fully Decoupled Teleop ===")
    print(" [POSITION] W/S (X), A/D (Y), Q/E (Z)  -> Solved via IK (Joints 0,1,2)")
    print(" [PITCH]    I/K                        -> Direct Joint Control (Joint 3)")
    print(" [ROLL]     U/O                        -> Direct Joint Control (Joint 4)")
    print(" [GRIPPER]  SPACE (Close), C (Open)    -> Direct Joint Control (Joint 5)")
    print(" [MODIFIER] Hold SHIFT for ultra-precision mode")
    print(" ESC to Quit")
    print("====================================")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            if keyboard.is_pressed('esc'):
                break

            speed_mult = 0.2 if keyboard.is_pressed('shift') else 1.0

            dx = np.zeros(3) 
            pitch_cmd = 0.0
            roll_cmd = 0.0
            gripper_cmd = 0.0
            
            # Position Inputs (Cartesian)
            if keyboard.is_pressed('w'): dx[0] += base_ik_step * speed_mult
            if keyboard.is_pressed('s'): dx[0] -= base_ik_step * speed_mult
            if keyboard.is_pressed('a'): dx[1] += base_ik_step * speed_mult
            if keyboard.is_pressed('d'): dx[1] -= base_ik_step * speed_mult
            if keyboard.is_pressed('q'): dx[2] += base_ik_step * speed_mult
            if keyboard.is_pressed('e'): dx[2] -= base_ik_step * speed_mult

            # Pitch Input (Joint Space - Joint 3)
            if keyboard.is_pressed('i'): pitch_cmd -= base_pitch_step * speed_mult
            if keyboard.is_pressed('k'): pitch_cmd += base_pitch_step * speed_mult

            # Roll Input (Joint Space - Joint 4)
            if keyboard.is_pressed('u'): roll_cmd += base_roll_step * speed_mult
            if keyboard.is_pressed('o'): roll_cmd -= base_roll_step * speed_mult

            # Gripper Input (Joint Space - Joint 5)
            if keyboard.is_pressed('space'): gripper_cmd += base_gripper * speed_mult
            if keyboard.is_pressed('c'):     gripper_cmd -= base_gripper * speed_mult

            # 1. Solve IK strictly for X/Y/Z translation using joints 0, 1, 2
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 0:
                jacp = np.zeros((3, env.model.nv))
                mujoco.mj_jacSite(env.model, env.data, jacp, None, site_id)
                
                # Slice Jacobian to ONLY use Base, Lift, and Elbow
                J_pos = jacp[:, :3] 
                lambda_sq = damping ** 2
                
                # Damped Least Squares
                J_dls = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T + lambda_sq * np.eye(3))
                delta_q = J_dls @ dx
                
                target_q[:3] += delta_q

                # Anti-Windup specifically for the 3 IK joints
                actual_q = env.data.qpos[:3]
                target_q[:3] = np.clip(target_q[:3], actual_q - max_lead, actual_q + max_lead)

            # 2. Apply Direct Joint Commands (Pitch = idx 3, Roll = idx 4, Gripper = idx 5)
            target_q[3] += pitch_cmd
            target_q[4] += roll_cmd
            target_q[5] += gripper_cmd

            # 3. Enforce Physical Limits globally
            target_q = np.clip(target_q, lower_limits, upper_limits)

            # Step physics
            env.step(target_q)
            viewer.sync()

    env.close()

if __name__ == "__main__":
    main()