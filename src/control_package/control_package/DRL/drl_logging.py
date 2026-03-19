# Third-party: ROS2
from copy import error

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.duration import Duration
from tf2_ros import *
from tf2_geometry_msgs import do_transform_pose
from rclpy.utilities import remove_ros_args

# Third-party: ROS2 Messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from custom_msgs.srv import LoggerRequest
from custom_msgs.msg import DRLLogging


# Python
import os
import sys
import numpy as np
import pandas as pd
import datetime
import argparse
from typing import Optional


#Custom
from base_package.manager import TransformManager

class LoggerNode(Node):
    def __init__(self, success_log_file_path: str=None, trajectory_log_file_path: str=None, object_log_file_path: str=None, episode_num: int=None):
        super().__init__("logger_node")
        self.get_logger().info("LoggerNode has been initialized.")
        self.setup_services()

        self.__success_log_file_path = success_log_file_path
        self.__trajectory_log_file_path = trajectory_log_file_path
        self.__object_log_file_path = object_log_file_path
        self.__episode_num = episode_num

        self._target_marker_id = 1  # Assuming the target marker has ID 1, adjust as needed
        self._transform_manager = TransformManager(self)

        self.__start_time: Optional[datetime.datetime] = None
        self.__end_time: Optional[datetime.datetime] = None
        self._running: bool = False
        self._sweep_direction: int = None

    
        if self.__success_log_file_path is None or self.__trajectory_log_file_path is None or self.__object_log_file_path is None:
            raise Exception("Log file paths must be provided when logging is enabled.")
        # Initialize log files with headers
        if os.path.exists(self.__trajectory_log_file_path):
            raise Exception(f"Trajectory log file already exists: {self.__trajectory_log_file_path}")
        if os.path.exists(self.__object_log_file_path):
            raise Exception(f"Object log file already exists: {self.__object_log_file_path}")

        self._gt_start_pos: np.ndarray = None
        self._gt_goal_pos: np.ndarray = None
        self._gt_pos: np.ndarray = None
        self._tcp_position: np.ndarray = None
        self._traj_tcp: pd.DataFrame = pd.DataFrame(columns=["tcp_x", "tcp_y", "tcp_z"])
        self._traj_obj: pd.DataFrame = pd.DataFrame(columns=["obj_x", "obj_y"])
        self._target_object_width: float = None

        #Subscription
        self._gt_callback = self.create_subscription(
            MarkerArray,
            "/natnet_client_node/marker_array",
            self._GT_callback,
            qos_profile=qos_profile_system_default
        )
        self._tcp_callback = self.create_subscription(
            PoseStamped,
            "/drl_logging",
            self._TCP_callback,
            qos_profile=qos_profile_system_default
        )

    def setup_services(self):
        self.srv = self.create_service(
            LoggerRequest,
            'logger_request',
            self.logger_request_callback
        )
        self.get_logger().info("LoggerRequest service is ready.")

    def logger_request_callback(self, request, response):
        self.get_logger().info(f"Received logger request: {request}")

        if request.start_logging:
            self._running = True
            self._sweep_direction = request.direction
            self.get_logger().info("Logger started.")
            self.get_logger().info(f"Sweep direction set to: {self._sweep_direction}")
            self.__start_time = datetime.datetime.now()
        else:
            self._running = False
            self.get_logger().info("Logger stopped.")
            self.__end_time = datetime.datetime.now()
            if self.__start_time and self.__end_time:
                duration = self.__end_time - self.__start_time
                self.get_logger().info(f"Logging duration: {duration}")
                error = np.linalg.norm(self._gt_goal_pos[:2] - self._gt_pos[:2])
                data = {
                    "Episode": self.__episode_num,
                    "Start_pos.x": self._gt_start_pos[0],
                    "Start_pos.y": self._gt_start_pos[1],
                    "Goal_pos.x": self._gt_goal_pos[0],
                    "Goal_pos.y": self._gt_goal_pos[1],
                    "End_pos.x": self._gt_pos[0],
                    "End_pos.y": self._gt_pos[1],
                    "Error": error,
                    "start_time": self.__start_time,
                    "End_time": self.__end_time,
                    "Elapsed_time": duration.total_seconds(),
                }

                if os.path.exists(self.__success_log_file_path):
                    existed_df = pd.read_csv(self.__success_log_file_path)
                    new_df = pd.DataFrame([data])
                    combined_df = pd.concat([existed_df, new_df], ignore_index=True)
                    combined_df.to_csv(self.__success_log_file_path, index=False)
                else:
                    pd.DataFrame([data]).to_csv(self.__success_log_file_path, mode='a', header=not os.path.exists(self.__success_log_file_path), index=False)
                self._traj_tcp.to_csv(self.__trajectory_log_file_path, index=False)
                self._traj_obj.to_csv(self.__object_log_file_path, index=False)
                
            else:
                raise ValueError("Start time or end time is not set.")

        response.success = True
        response.message = "Logger state updated successfully."
        return response
    
    def _GT_callback(self, msg: MarkerArray):
        for marker in msg.markers:
            marker: Marker

            # Check if the marker is the target marker
            if marker.id == self._target_marker_id:
                pos = PoseStamped(
                    header=Header(
                        frame_id=marker.header.frame_id,
                        stamp=self.get_clock().now().to_msg(),
                    ),
                    pose=marker.pose,
                )
                pose_in_base: PoseStamped = self._transform_manager.transform_pose(
                    pose=pos,
                    target_frame="base_link",
                    source_frame=marker.header.frame_id,
                )

                if pose_in_base is None:
                    self.get_logger().warn(
                        f"Failed to transform marker pose from {marker.header.frame_id} to base_link."
                    )
                    return

                if self._gt_start_pos is None and self._sweep_direction is not None:
                    self._gt_start_pos = np.array(
                        [
                            pose_in_base.pose.position.x,
                            pose_in_base.pose.position.y,
                        ],
                        dtype=np.float32,
                    )

                    if self._sweep_direction == 0:

                        self._gt_goal_pos = np.array(
                            [
                                pose_in_base.pose.position.x,
                                pose_in_base.pose.position.y - 0.18,
                                1.05 - 0.79505,
                            ],
                            dtype=np.float32,
                        )
                    elif self._sweep_direction == 1:
                        self._gt_goal_pos = np.array(
                            [
                                pose_in_base.pose.position.x,
                                pose_in_base.pose.position.y + 0.18,
                                1.05 - 0.79505,
                            ],
                            dtype=np.float32,   
                        )
                    else:
                        raise Exception(f"Unknown sweep direction: {self._sweep_direction}")

                    self.get_logger().info(f"GT Start Position: {self._gt_start_pos}, GT Goal Position: {self._gt_goal_pos}")
                    
                self._gt_pos = np.array(
                    [
                        pose_in_base.pose.position.x,
                        pose_in_base.pose.position.y,
                    ],
                    dtype=np.float32,
                )

                break

    def _TCP_callback(self, msg: PoseStamped):
        if not self._running:
            return
        
        self._tcp_position = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)


    def logging(self):
        if self._running:
            # Log TCP trajectory
            if self._tcp_position is not None:
                self._traj_tcp = pd.concat([self._traj_tcp, pd.DataFrame([{"tcp_x": self._tcp_position[0], "tcp_y": self._tcp_position[1], "tcp_z": self._tcp_position[2]}])], ignore_index=True)
            # Log object trajectory
            if self._gt_pos is not None:
                self._traj_obj = pd.concat([self._traj_obj, pd.DataFrame([{"obj_x": self._gt_pos[0], "obj_y": self._gt_pos[1]}])], ignore_index=True)


def parse_args():
    """ROS2 인자를 제거한 뒤 argparse로 파싱"""
    filtered_args = remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser(description="Pose Estimate Node")
    parser.add_argument(
        "--episode_num",
        type=int,
        default=1,
        help="Episode number for logging purposes."
    )
    return parser.parse_args()  # [1:] to skip script name


def main():
    args = parse_args()
    rclpy.init(args=None)
    node = LoggerNode(success_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_success_log.csv",
                       trajectory_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_trajectory_log0319_{args.episode_num}.csv",
                       object_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_object_log0319_{args.episode_num}.csv",
                       episode_num=args.episode_num)

    import threading
    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    r = node.create_rate(100)
    while rclpy.ok():
        # node.get_logger().info("LoggerNode is running...")
        node.logging()
        r.sleep()
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()