import genesis as gs
import numpy as np
import torch
import h5py

######################## 工具函数 ########################
def to_cpu_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)

######################## 初始化 ########################
gs.init(backend=gs.gpu)

scene = gs.Scene(
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(3, -1, 1.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=30,
        max_FPS=60,
    ),
    sim_options=gs.options.SimOptions(dt=0.01),
    show_viewer=True,
)
cam = scene.add_camera(
    res=(640, 480),                # 分辨率
    pos=(1.0, 0.0, 0.6),           # 相机位置
    lookat=(0.65, 0.0, 0.0),       # 看向物体
    fov=60,
    GUI=False,                     # 不弹独立窗口
)
plane = scene.add_entity(gs.morphs.Plane())
cube  = scene.add_entity(gs.morphs.Box(size=(0.04, 0.04, 0.04), pos=(0.65, 0.0, 0.02)))
franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))


scene.build()

motors_dof  = np.arange(7)       # 关节电机
fingers_dof = np.arange(7, 9)    # 夹爪 DOF
end_effector = franka.get_link("hand")  # hand link 的索引

# 设置控制增益
franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]))
franka.set_dofs_kv(np.array([ 450,  450,  350,  350,  200,  200,  200,  10,  10]))
franka.set_dofs_force_range(
    np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    np.array([ 87,  87,  87,  87,  12,  12,  12,  100, 100]),
)

######################## HDF5 文件初始化 ########################
num_dofs = franka.n_dofs
h5file = h5py.File("grasp_demo.hdf5", "w")
qpos_ds  = h5file.create_dataset("/observation/qpos",     (0, num_dofs), maxshape=(None, num_dofs))
ee_ds    = h5file.create_dataset("/observation/ee_pose",  (0, 7), maxshape=(None, 7))
obj_ds   = h5file.create_dataset("/observation/obj_pose", (0, 7), maxshape=(None, 7))
rgb_ds   = h5file.create_dataset("/observation/images/head_rgb",      (0, 480, 640, 3), maxshape=(None, 480, 640, 3), dtype=np.uint8)
depth_ds = h5file.create_dataset("/observation/images/head_depth",    (0, 480, 640),    maxshape=(None, 480, 640),    dtype=np.float32)

def append_data():
    # ---- 机器人数据 ----
    qpos = to_cpu_np(franka.get_qpos())
    ee_pos  = to_cpu_np(end_effector.get_pos())
    ee_quat = to_cpu_np(end_effector.get_quat())
    ee_pose = np.concatenate([ee_pos, ee_quat])
    obj_qpos = to_cpu_np(cube.get_qpos())

    # ---- 相机数据 ----
    rgb_arr, depth_arr, _, _ = cam.render(rgb=True, depth=True)
    rgb   = to_cpu_np(rgb_arr)
    depth = to_cpu_np(depth_arr)

    # ---- 写入 HDF5 ----
    for ds, data in zip([qpos_ds, ee_ds, obj_ds], [qpos, ee_pose, obj_qpos]):
        ds.resize(ds.shape[0] + 1, axis=0)
        ds[-1] = data

    rgb_ds.resize(rgb_ds.shape[0] + 1, axis=0)
    rgb_ds[-1] = (rgb * 255).astype(np.uint8)

    depth_ds.resize(depth_ds.shape[0] + 1, axis=0)
    depth_ds[-1] = depth.astype(np.float32)
######################## 动作流程 ########################

# 打开夹爪
q = franka.get_dofs_position()
q[-2:] = 0.04
franka.control_dofs_position(q)
for _ in range(50):
    scene.step()
    append_data()

# 移动到物体上方
q_target = franka.inverse_kinematics(link=end_effector,
                                     pos=np.array([0.65, 0.0, 0.25]),
                                     quat=np.array([0, 1, 0, 0]))
q_target = to_cpu_np(q_target)
q_target[-2:] = 0.04

path = franka.plan_path(qpos_goal=q_target, num_waypoints=200)
path_np = [to_cpu_np(wp) for wp in path]
for wp in path_np:
    franka.control_dofs_position(wp)
    scene.step()
    append_data()

# 接近物体
q_target = franka.inverse_kinematics(link=end_effector,
                                     pos=np.array([0.65, 0.0, 0.13]),
                                     quat=np.array([0, 1, 0, 0]))
q_target = to_cpu_np(q_target)
franka.control_dofs_position(q_target[:-2], motors_dof)
for _ in range(100):
    scene.step()
    append_data()

# 抓取
franka.control_dofs_position(q_target[:-2], motors_dof)
franka.control_dofs_force(np.array([-0.5, -0.5]), fingers_dof)
for _ in range(120):
    scene.step()
    append_data()

# 抬升
q_target = franka.inverse_kinematics(link=end_effector,
                                     pos=np.array([0.65, 0.0, 0.28]),
                                     quat=np.array([0, 1, 0, 0]))
q_target = to_cpu_np(q_target)
franka.control_dofs_position(q_target[:-2], motors_dof)
for _ in range(200):
    scene.step()
    append_data()

######################## 保存文件 ########################
h5file.close()
print("✅ 数据已保存到 grasp_demo.hdf5")