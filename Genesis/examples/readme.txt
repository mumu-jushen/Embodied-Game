最终格式
.
├─meta 
│ ├─episodes.jsonl
│ ├─modality.json # -> GR00T LeRobot specific
│ ├─info.json
│ └─tasks.jsonl
├─videos
│ └─chunk-000
│   └─observation.images.ego_view  # -> ego_view  是自定义摄像头的名字 
│     └─episode_000001.mp4
│     └─episode_000000.mp4
  └─chunk-000
    ├─episode_000001.parquet
    └─episode_000000.parquet
___________________________________________________________________________________________________________________________________________

info.json
{
  "dataset_name": "LeRobot PushT",
  "author": "Your Name",
  "date": "2025-08-26",
  "description": "机器人推物任务数据集，包含多模态传感器数据和视频。",
  "num_episodes": 100,
  "fps": 30,
  "modality": ["left_arm", "right_arm", "head", "waist", "left_hand", "right_hand", "velocity_left_arm", "velocity_right_arm", "velocity_head", "velocity_waist", "force_left_tcp", "force_right_tcp"]
}

tasks_data = {}
tasks_data["dataset_name"] = [0]
tasks_data["author"] = [0]
tasks_data["task_index"] = [0]
tasks_data["description"] = [0]
tasks_data["num_episodes"] = [0]

episode_data = {
    "episode_index": hdf5_files.index(hdf5_file_name),
    "task_index": 3,
    "start_frame": 0,
    "end_frame": frame_count - 1,
    "duration": round((frame_count - 1) / fps, 2)
    }

同一个文件夹不能运行两次，因为jsonl用的是追加模式，每次都会把当时数据加进去