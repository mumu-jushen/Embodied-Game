import os
import h5py
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import cv2
import av
from datetime import date
#---    先创好lerobot文件夹
base_dir = os.path.join(os.getcwd(), 'output')
os.makedirs(os.path.join(base_dir, 'meta'), exist_ok=True)

os.makedirs(os.path.join(base_dir, 'videos', 'chunk-000','head_left_rgb'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'videos', 'chunk-000','head_right_rgb'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'videos', 'chunk-000','chest_rgb'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'videos', 'chunk-000','chest_depth'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'videos'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'data'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'data', 'chunk-000'), exist_ok=True)


#---    文件夹地址
meta_dir = os.path.join(base_dir, 'meta')

videos_dir = os.path.join(base_dir, 'videos')
videos_chunk_000_dir = os.path.join(videos_dir, 'chunk-000')
videos_chunk_000_head_left_rgb = os.path.join(videos_chunk_000_dir, 'head_left_rgb')
videos_chunk_000_head_right_rgb = os.path.join(videos_chunk_000_dir, 'head_right_rgb')
videos_chunk_000_chest_rgb = os.path.join(videos_chunk_000_dir, 'chest_rgb')
videos_chunk_000_chest_depth = os.path.join(videos_chunk_000_dir, 'chest_depth')

data_dir = os.path.join(base_dir, 'data')
data_chunk_000_dir = os.path.join(data_dir, 'chunk-000')
#---    导入hdf5的文件夹
hdf5_file_path = '/home/rxjqr/Genesis/examples/logs'


#---    配置文件
fps = 30
hdf5_files = sorted(os.listdir(hdf5_file_path))
Arthur = "ll"
Datesets_name = "Lerobot"

#---    提取hdf5文件，处理
for hdf5_file_name in tqdm(hdf5_files, desc="Processing HDF5 files", unit="file"):
    num_episodes = len(hdf5_files)
    file_path = os.path.join(hdf5_file_path, hdf5_file_name)
    # 处理视频数据
    with h5py.File(os.path.join(hdf5_file_path, hdf5_file_name), 'r') as root:
        streams = {                                                     #这里的文件见遥操数据采集说明
        "rgb_ds"   : root["/observation/images/head_rgb"],
        "depth_ds" : root["/observation/images/head_depth"]
        }

        avi_writers = {}
        mp4_writers = {}

        video_idx = hdf5_files.index(hdf5_file_name)
        video_name = f"episode_{video_idx:06d}.mp4"
        depth_video_name = f"episode_{video_idx:06d}.avi"
        # --- RGB 视频写入器 ---                                   
        for name, frames in streams.items():
            if name == "rgb_ds":
                h, w, _ = frames[0].shape
                mp4_path = os.path.join(videos_chunk_000_head_left_rgb, video_name)
                mp4_writers[name] = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            elif name == "rgb2":
                h, w, _ = frames[0].shape
                mp4_path = os.path.join(videos_chunk_000_head_right_rgb, video_name)
                mp4_writers[name] = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            elif name == "rgb3":
                h, w, _ = frames[0].shape
                mp4_path = os.path.join(videos_chunk_000_chest_rgb, video_name)
                mp4_writers[name] = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        # --- Depth 视频写入器 (FFV1) ---
        depth_frames = streams['depth_ds'][:].astype(np.uint16)
        frame_count, h_d, w_d = depth_frames.shape
        depth_path = os.path.join(videos_chunk_000_chest_depth, depth_video_name)
        container = av.open(depth_path, mode='w')
        stream = container.add_stream('ffv1', rate=fps)
        stream.width = w_d
        stream.height = h_d
        stream.pix_fmt = 'gray16le'


        # --- 写入帧 ---
        for i in tqdm(range(len(streams['depth_ds'])), desc=f"Writing frames {hdf5_file_name}", leave=False, unit="frame"):  #以rgb1为基准，遍历所有的流
            for name, frames in streams.items():
                if name in ["rgb_ds"]:
                    frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
                    mp4_writers[name].write(frame_bgr)
        # --- 达成效果，把rgb转化成bgr ---
            # Depth
            frame_depth = depth_frames[i]  #取出第I帧，得到一个二维图像
            video_frame = av.VideoFrame.from_ndarray(frame_depth, format='gray16le')
            for packet in stream.encode(video_frame):
                container.mux(packet)
    # --- 写episodes.jsonl
    episode_data = {
    "episode_index": hdf5_files.index(hdf5_file_name),
    "task_index": 3,
    "start_frame": 0,
    "end_frame": frame_count - 1,
    "duration": round((frame_count - 1) / fps, 2)
    }
    jsonl_dir = os.path.join(meta_dir)
    os.makedirs(jsonl_dir, exist_ok=True)
    jsonl_path = os.path.join(jsonl_dir, 'episodes.jsonl')
    with open(jsonl_path, 'a', encoding='utf-8') as fjson:    #这里是追加模式，可能会有坑，就是运行两次的时候会把
        fjson.write(json.dumps(episode_data, ensure_ascii=False) + '\n')
    #处理除视频外的数据
    with h5py.File(file_path, 'r') as f:
        qpos_ds  = f["/observation/qpos"]
        ee_ds    = f["/observation/ee_pose"]
        obj_ds   = f["/observation/obj_pose"]

        # 合并，axis=1 表示按列拼接
        angle_arms = np.hstack([
            qpos_ds, ee_ds, obj_ds
        ])

        # 转成 DataFrame
        df_arms = pd.DataFrame(angle_arms)
        episode_idx = hdf5_files.index(hdf5_file_name)
        parquet_name = f"episode_{episode_idx:06d}.parquet"
        df_arms.to_parquet(os.path.join(data_chunk_000_dir, parquet_name))
        
            
        # 自动生成索引范围并保存为 JSON 文件 注意，这里为modality.json
        idx = 0
        modality = {}
        modality["state"] = {}

        modality_names = [
            ("qpos", qpos_ds),
            ("ee_pose", ee_ds),
            ("obj_pose", obj_ds),
        ]

        for name, arr in modality_names:
            num = arr.shape[1]
            modality["state"][name] = {"start": idx, "end": idx + num - 1}
            idx += num

        with open(os.path.join(meta_dir, 'modality.json'), 'w', encoding='utf-8') as fjson:  # 在当前目录下新建（或覆盖）一个名为 modality.json 的文件，并用 UTF-8 编码写入。
            json.dump(modality, fjson, ensure_ascii=False, indent=4)
#数据处理完后，写两个总的json文件，info.json和tasks.jsonl
info_data = {}
info_data["dataset_name"] = ["LeRobot"]
info_data["author"] = [Arthur]
info_data["data"] = [str(date.today())]
info_data["description"] = ["grasp cube"]
info_data["num_episodes"] = [num_episodes]
info_data["fps"] = [fps]
info_data["modality"] = [
    "qpos", "ee_pose", "obj_pose"
]
with open(os.path.join(meta_dir,'info.json'), 'w', encoding='utf-8') as fjson:  # 在当前目录下新建（或覆盖）一个名为 info.json 的文件，并用 UTF-8 编码写入。
    json.dump(info_data, fjson, ensure_ascii=False, indent=4)
tasks_data = {}
tasks_data["dataset_name"] = "LeRobot"
tasks_data["author"] = Arthur
tasks_data["task_index"] = 0
tasks_data["description"] = "grasp cube"
tasks_data["num_episodes"] = num_episodes

tasks_jsonl_dir = os.path.join(meta_dir)
os.makedirs(tasks_jsonl_dir, exist_ok=True)
tasks_jsonl_path = os.path.join(tasks_jsonl_dir, 'tasks.jsonl')
with open(tasks_jsonl_path, 'a', encoding='utf-8') as fjson:    #这里是追加模式，可能会有坑，就是运行两次的时候会把
    fjson.write(json.dumps(tasks_data, ensure_ascii=False) + '\n')