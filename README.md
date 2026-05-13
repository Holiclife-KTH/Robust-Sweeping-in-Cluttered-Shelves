# Robust Sweeping in Cluttered Shelves: Contact-Aware Reward Desing and Physics-Tuned Sim-to-Real Transfer


[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![ROS 2](https://img.shields.io/badge/ROS%202-Humble-green)](https://docs.ros.org/en/humble/)
[![C++](https://img.shields.io/badge/C++-17-red)](https://cplusplus.com/)

------
## 1. Project Structure

```text
src/
  base_package/              # 공통 매니저, TF/이미지/포인트클라우드 유틸, launch
  control_package/           # DRL/UNO push 기반 제어 로직
  custom_msgs/               # BBox/BBox3D 등 커스텀 메시지/서비스
  perception_package/        # segmentation, pose_estimate
  ur5e_bringup/              # MoveIt2 설정/launch
  urdf_description/          # UR/Robotiq description
  ROS2_Helios2_RGB_KIT/      # Lucid Helios/Triton camera driver
```


## Prerequisites
### Universal robot driver (Humble)
```
sudo apt install ros-humble-ur<img width="930" height="748" alt="image (4)" src="https://github.com/user-attachments/assets/5b2f1a5a-5ee3-4ceb-93a0-2af137120333" />

```
### Moveit2
```
sudo apt install ros-humble-moveit
```
### Arena SDK



## Abstract
\\
\\
---
## Run
### Workspace activation
```
cd ~/workspace/Robust-Sweeping-in-Cluttered-Shelves
source install/setup.bash
```

### Universal driver
```
ros2 launch ur_robot_driver ur5e.launch.py robot_ip:=192.168.3.2 use_tool_communication:=true tool_voltage:=24
```

### Static TFs
```
ros2 launch base_package static_tf.launch.py
```

### Lucid Camera node (RGB, Pointcloud)
```
ros2 run lucid_camera_node lucid_node
```

### Perception (Segmentation, Pose estimation)
```
python3 -m perception_package.segmentation
python3 -m perception_package.pose_estimate --target_cls {TARGET_CLASS}
```

<img width="1145" height="610" alt="image (5)" src="https://github.com/user-attachments/assets/8031796b-8505-453e-8060-ef363745a90e" />
<img width="785" height="609" alt="image (3)" src="https://github.com/user-attachments/assets/1b9d19bb-60fb-4059-b7bf-76fc8a7ba4cd" />
<img width="930" height="748" alt="image (4)" src="https://github.com/user-attachments/assets/9f6231d7-705c-4623-85b7-104d9f9262b4" />


### Control Node (DRL, UNO-Push, HITL)
```
python3 -m control_package.DRL.run_task_with_vision #DRL Based
```


