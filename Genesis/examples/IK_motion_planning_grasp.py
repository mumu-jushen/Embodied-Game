import json
import numpy as np
# ---------------- 四元数工具（XYZW, w 在最后） ----------------
def q_normalize_xyzw(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([0, 0, 0, 1], dtype=np.float64)
def q_mul_xyzw(q1, q2):
    # (x, y, z, w)
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2
    ], dtype=np.float64)
def q_conj_xyzw(q):
    x, y, z, w = q
    return np.array([-x, -y, -z,  w], dtype=np.float64)
def q_rotate_vec_xyzw(q, v):
    # 用四元数旋转向量
    q = q_normalize_xyzw(q)
    vx, vy, vz = v
    # 把 v 视作纯虚四元数 (x, y, z, 0)
    vq = np.array([vx, vy, vz, 0.0], dtype=np.float64)
    return q_mul_xyzw(q_mul_xyzw(q, vq), q_conj_xyzw(q))[:3]
def compose_pose_xyzw(pos_ref, quat_ref, pos_rel, quat_rel):
    # 绝对位姿 = 参考位姿 ∘ 相对位姿
    pos_abs  = pos_ref + q_rotate_vec_xyzw(quat_ref, pos_rel)
    quat_abs = q_mul_xyzw(quat_ref, quat_rel)
    return pos_abs, q_normalize_xyzw(quat_abs)
# ---------------- 读取你给的 JSON（示例：从字符串或文件） ----------------
# 1) 如果你已读成 dict（比如 data = json.loads(...)），直接用：
# data = {...}  # 就是你发的那份 JSON
# 2) 或者从文件读取：
with open("grasp_rel_poses.json", "r") as f:
    data = json.load(f)
rel_pos  = np.array(data["rel_pos"], dtype=np.float64)
rel_quat = q_normalize_xyzw(np.array(data["rel_quat"], dtype=np.float64))  # [x,y,z,w]


import genesis as gs
gs.init(backend=gs.gpu)

########################## create a scene ##########################
scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(
        dt=0.01,
    ),
    show_viewer=True,
)

########################## entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)
cube = scene.add_entity(
    gs.morphs.Box(
        size=(0.04, 0.04, 0.04),
        pos=(0.65, 0.0, 0.02),
    )
)
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

# 添加相机
cam = scene.add_camera(
    res=(640, 480),
    pos=(1.0, 0.0, 0.6),
    lookat=(0.65, 0.0, 0.0),
    fov=60,
    GUI=False,
)
########################## build ##########################
scene.build()

motors_dof = np.arange(7)
fingers_dof = np.arange(7, 9)

# set control gains
franka.set_dofs_kp(
    np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
)
franka.set_dofs_kv(
    np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
)
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
)

end_effector = franka.get_link("hand")



# ---------------- 获取当前物体（cube）的世界位姿 ----------------
# try:
print("cube.get_qpos()=", cube.get_qpos())

# 从 cube 获取 qpos (cuda tensor)
qpos = cube.get_qpos().detach().cpu().numpy()  # [x, y, z, qw, qx, qy, qz]
# 拆分平移和四元数
cube_pos  = qpos[:3]
cube_quat_wxyz = qpos[3:]  # [qw, qx, qy, qz]
# 如果后面函数要求 [x, y, z, w] 顺序，就转一下：
cube_quat_xyzw = np.array([cube_quat_wxyz[1], cube_quat_wxyz[2], cube_quat_wxyz[3], cube_quat_wxyz[0]])
cube_quat = q_normalize_xyzw(cube_quat_xyzw)

print(f"cube_pos={cube_pos}, cube_quat={cube_quat}")

# ---------------- 计算末端执行器的世界目标位姿（由相对位姿合成） ----------------
target_pos, target_quat = compose_pose_xyzw(cube_pos, cube_quat, rel_pos, rel_quat)  # XYZW
# ---------------- 打开夹爪（保持与原脚本一致） ----------------
q = franka.get_dofs_position()
q[-2:] = 0.04
franka.control_dofs_position(q)
for _ in range(30):
    scene.step()
# ---------------- 规划到目标位姿并执行 ----------------
q_target = franka.inverse_kinematics(link=end_effector, pos=target_pos, quat=target_quat)  # 注意：Genesis 用 XYZW
q_target = q_target.detach().cpu().numpy().astype(np.float64, copy=False)
q_target[-2:] = 0.04  # 到位时保持张开
path = franka.plan_path(qpos_goal=q_target, num_waypoints=200,ignore_collision=False,resolution=0.05, max_retry=5,smooth_path=True)
dbg = scene.draw_debug_path(path, franka)
for wp in path:
    franka.control_dofs_position(wp)
    scene.step()
scene.clear_debug_object(dbg)
for _ in range(80):
    scene.step()
# ---------------- 抓取：锁定 7 个关节位姿，给夹爪力闭合 ----------------
motors_dof  = np.arange(7)
fingers_dof = np.arange(7, 9)
franka.control_dofs_position(q_target[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5], dtype=np.float64), fingers_dof)
for _ in range(120):
    scene.step()
# ---------------- 沿世界 Z 轴抬升 10cm ----------------
lift_pos  = target_pos + np.array([0.0, 0.0, 0.10], dtype=np.float64)
lift_quat = target_quat  # 姿态保持不变
q_lift = franka.inverse_kinematics(link=end_effector, pos=lift_pos, quat=lift_quat)
franka.control_dofs_position(q_lift[:-2], motors_dof)
for _ in range(200):
    scene.step()