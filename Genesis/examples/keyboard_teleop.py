"""
Keyboard Controls:
↑	- Move Forward (North)
↓	- Move Backward (South)
←	- Move Left (West)
→	- Move Right (East)
n	- Move Up
m	- Move Down
j	- Rotate Counterclockwise
k	- Rotate Clockwise
u	- Reset Scene
space	- Press to close gripper, release to open gripper
esc	- Quit
"""
import os
import random
import threading
import json


import genesis as gs
import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R


class KeyboardDevice:
    def __init__(self):
        self.pressed_keys = set()
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)

    def start(self):
        self.listener.start()

    def stop(self):
        self.listener.stop()
        self.listener.join()

    def on_press(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.add(key)

    def on_release(self, key: keyboard.Key):
        with self.lock:
            self.pressed_keys.discard(key)

    def get_cmd(self):
        return self.pressed_keys


def build_scene():
    ########################## init ##########################
    gs.init(seed=0, precision="32", logging_level="info", backend=gs.cpu)
    np.set_printoptions(precision=7, suppress=True)

    ########################## create a scene ##########################
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=4,
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.02,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
        show_FPS=False,
    )

    ########################## entities ##########################
    entities = dict()
    entities["plane"] = scene.add_entity(
        gs.morphs.Plane(),
    )

    entities["robot"] = scene.add_entity(
        material=gs.materials.Rigid(gravity_compensation=1),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda.xml",
            euler=(0, 0, 0),
        ),
    )
    entities["cube"] = scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
    )

    entities["target"] = scene.add_entity(
        gs.morphs.Mesh(
            file="meshes/axis.obj",
            scale=0.15,
            collision=False,
        ),
        surface=gs.surfaces.Default(color=(1, 0.5, 0.5, 1)),
    )

    ########################## build ##########################
    scene.build()

    return scene, entities


def run_sim(scene, entities, clients):
    robot = entities["robot"]
    target_entity = entities["target"]

    robot_init_pos = np.array([0.5, 0, 0.55])
    robot_init_R = R.from_euler("y", np.pi)
    target_pos = robot_init_pos.copy()
    target_R = robot_init_R

    n_dofs = robot.n_dofs
    motors_dof = np.arange(n_dofs - 2)
    fingers_dof = np.arange(n_dofs - 2, n_dofs)
    ee_link = robot.get_link("hand")
    # === grasp logs ===
    grasp_logs = []  # 每条: dict(time, ee_pos, ee_quat, obj_pos, obj_quat, rel_pos, rel_quat)
    # === helper: 将当前EE与物体的相对姿态(EE→Obj)算出来 ===
    def compute_ee_to_obj_relative():
        # EE全局位姿
        ee_pos = ee_link.get_pos()            # shape (3,)
        ee_quat = ee_link.get_quat()          # scalar_first=True 四元数 (w, x, y, z)
        # 物体全局位姿
        obj = entities["cube"]
        obj_pos = obj.get_pos()
        obj_quat = obj.get_quat()
        # 相对旋转 R_rel = R_ee^{-1} * R_obj
        R_ee = R.from_quat([ee_quat[1], ee_quat[2], ee_quat[3], ee_quat[0]])   # 转为 (x,y,z,w)
        R_obj = R.from_quat([obj_quat[1], obj_quat[2], obj_quat[3], obj_quat[0]])
        R_rel = R_ee.inv() * R_obj
        rel_quat_xyzw = R_rel.as_quat()  # (x,y,z,w)
        rel_quat_wxyz = np.array([rel_quat_xyzw[3], rel_quat_xyzw[0], rel_quat_xyzw[1], rel_quat_xyzw[2]])
        # 相对平移 p_rel = R_ee^{-1} * (p_obj - p_ee)
        rel_pos = R_ee.inv().apply(np.array(obj_pos) - np.array(ee_pos))
        print(f"rel_pos={rel_pos}, rel_quat(wxyz)={rel_quat_wxyz}")
        return {
            "rel_pos": rel_pos.astype(float),
            "rel_quat": rel_quat_wxyz.astype(float),
        }
    def reset_scene():
        nonlocal target_pos, target_R
        target_pos = robot_init_pos.copy()
        target_R = robot_init_R
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat)
        robot.set_qpos(q[:-2], motors_dof)

        entities["cube"].set_pos((random.uniform(0.2, 0.4), random.uniform(-0.2, 0.2), 0.05))
        entities["cube"].set_quat(R.from_euler("z", random.uniform(0, np.pi * 2)).as_quat(scalar_first=True))

    print("\nKeyboard Controls:")
    print("↑\t- Move Forward (North)")
    print("↓\t- Move Backward (South)")
    print("←\t- Move Left (West)")
    print("→\t- Move Right (East)")
    print("n\t- Move Up")
    print("m\t- Move Down")
    print("j\t- Rotate Counterclockwise")
    print("k\t- Rotate Clockwise")
    print("u\t- Reset Scene")
    print("space\t- Press to close gripper, release to open gripper")
    print("esc\t- Quit")

    # reset scen before starting teleoperation
    reset_scene()

    # start teleoperation
    stop = False
    while not stop:
        pressed_keys = clients["keyboard"].pressed_keys.copy()

        # reset scene:
        reset_flag = False
        reset_flag |= keyboard.KeyCode.from_char("u") in pressed_keys
        if reset_flag:
            reset_scene()

        # stop teleoperation
        stop = keyboard.Key.esc in pressed_keys

        # get ee target pose
        is_close_gripper = False
        dpos = 0.002
        drot = 0.01
        for key in pressed_keys:
            if key == keyboard.Key.up:
                target_pos[0] -= dpos
            elif key == keyboard.Key.down:
                target_pos[0] += dpos
            elif key == keyboard.Key.right:
                target_pos[1] += dpos
            elif key == keyboard.Key.left:
                target_pos[1] -= dpos
            elif key == keyboard.KeyCode.from_char("n"):
                target_pos[2] += dpos
            elif key == keyboard.KeyCode.from_char("m"):
                target_pos[2] -= dpos
            elif key == keyboard.KeyCode.from_char("j"):
                target_R = R.from_euler("z", drot) * target_R
            elif key == keyboard.KeyCode.from_char("k"):
                target_R = R.from_euler("z", -drot) * target_R
            elif key == keyboard.Key.space:
                is_close_gripper = True

        # control arm
        target_quat = target_R.as_quat(scalar_first=True)
        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, err = robot.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, return_error=True)
        robot.control_dofs_position(q[:-2], motors_dof)
        # control gripper
        if is_close_gripper:
            robot.control_dofs_force(np.array([-1.0, -1.0]), fingers_dof)
            rec = compute_ee_to_obj_relative()
            grasp_logs = rec
        else:
            robot.control_dofs_force(np.array([1.0, 1.0]), fingers_dof)

        scene.step()

    # ===== write logs to JSON =====
    os.makedirs("logs", exist_ok=True)
    json_path = "logs/grasp_rel_poses.json"
    def np_convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.float32, np.float64)):
            return float(o)
        return o
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(grasp_logs, f, ensure_ascii=False, indent=2, default=np_convert)
    print(f"[LOG] Saved {len(grasp_logs)} grasp records to {json_path}")

def main():
    clients = dict()
    clients["keyboard"] = KeyboardDevice()
    clients["keyboard"].start()

    scene, entities = build_scene()
    run_sim(scene, entities, clients)


if __name__ == "__main__":
    main()
