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
sudo apt install ros-humble-ur
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

### Control Node (DRL, UNO-Push, HITL)
```
python3 -m control_package.DRL.run_task_with_vision #DRL Based
```


