# ROS2
import sys

import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time

from rclpy.qos import QoSProfile, qos_profile_system_default
import tf2_ros
from rclpy.utilities import remove_ros_args

# Message
from std_msgs.msg import *
from geometry_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from tf2_geometry_msgs import do_transform_pose
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from custom_msgs.msg import DRLLogging
from custom_msgs.srv import LoggerRequest

from control_package.DRL.ur import URSweepPolicy
from base_package.manager import TransformManager


# UR3
import rtde_control
import rtde_receive

# TF
from tf2_ros import *

# Python
import numpy as np
from enum import Enum

# import torch
import math
import time
import datetime
import argparse
import os
import sys
from loguru import logger
import pandas as pd

# UR3
import rtde_control
import rtde_receive


class Direction(Enum):
    RIGHT = 0
    LEFT = 1


class SweepPolicy(Node):
    """ROS2 node for controlling a UR robot's reach policy"""

    # Define simulation degree-of-freedom angle limits: (Lower limit, Upper limit, Inversed flag)
    SIM_DOF_ANGLE_LIMITS = [
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
        (-360, 360, False),
    ]

    # Define servo angle limits (in radians)
    PI = math.pi
    SERVO_ANGLE_LIMITS = [
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
        (-2 * PI, 2 * PI),
    ]

    # ROS topics and joint names
    STATE_TOPIC = "/scaled_joint_trajectory_controller/state"
    CMD_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
    JOINT_NAMES = [
        "elbow_joint",
        "shoulder_lift_joint",
        "shoulder_pan_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]

    # Mapping from joint name to simulation action index
    JOINT_NAME_TO_IDX = {
        "elbow_joint": 2,
        "shoulder_lift_joint": 1,
        "shoulder_pan_joint": 0,
        "wrist_1_joint": 3,
        "wrist_2_joint": 4,
        "wrist_3_joint": 5,
    }

    def __init__(self, fail_quietly: bool = False, 
                 verbose: bool = False, 
                 log: bool = False, 
                 episode_num: int = 1, 
                 success_log_file_path: str = None, 
                 trajectory_log_file_path: str = None, 
                 object_log_file_path: str = None):
        """Initialize the SweepPolicy node"""
        super().__init__("sweep_policy_node")

        self.robot = URSweepPolicy(
            model_path="/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/resource/260311"
        )

        # UR
        IP = "192.168.3.2"
        self.rtde_c = rtde_control.RTDEControlInterface(IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(IP)

        self.target_command = np.zeros(7)
        self.step_size = 1.0 / 100.0  # 10 ms period = 100 Hz

        self.i = 0
        self._episode_num = episode_num
        self._target_marker_id = 1  # NatNet에서 추적할 마커 ID
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        self.log = log
        self.episode_num = episode_num
        self.pub_freq = 100.0  # Hz
        self.current_pos = None  # Dictionary of current joint positions
        self._target_object_width:float=None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._transform_manager = TransformManager(node=self)
        self._sweep_direction = Direction.LEFT

        self._target_object_sub = self.create_subscription(
            PoseStamped,
            "/pose_estimate/position",
            self._target_object_position_callback,
            qos_profile=qos_profile_system_default,
        )
        self._target_object_width_sub = self.create_subscription(
            Float32,
            "/pose_estimate/width",
            self._target_object_width_callback,
             qos_profile=qos_profile_system_default,
        )

        self._target_pos: np.ndarray = None
        self._goal_pos: np.ndarray = None
    
        self._target_point_pub = self.create_publisher(
            PoseStamped,
            "/target_point",
            qos_profile=qos_profile_system_default,
        )
        self._drl_logging_pub = self.create_publisher(
            PoseStamped,
            "/drl_logging",
            qos_profile=qos_profile_system_default,
        )


        self.min_traj_dur = 0  # Minimum trajectory duration in seconds


        self.reset()

        self.get_logger().info("SweepPolicy node initialized.")

    def _target_object_width_callback(self, msg: Float32):
        # self.get_logger().info(f"Received target object width: {msg.data}")
        if msg is None:
            return
        if self._target_object_width is None:
            self._target_object_width = msg.data

    def _target_object_position_callback(self, msg: PoseStamped):
        pose_in_base: PoseStamped = self._transform_manager.transform_pose(
            pose=msg,
            target_frame="base_link",
            source_frame=msg.header.frame_id,
        )

        if pose_in_base is None:
            self.get_logger().warn(
                f"Failed to transform target object position from {msg.header.frame_id} to base_link."
            )
            return
        self._target_point_pub.publish(pose_in_base)

        if self._target_pos is None:

            if self._sweep_direction == Direction.RIGHT:
                self._goal_pos = np.array(
                    [
                        pose_in_base.pose.position.x,
                        pose_in_base.pose.position.y - 0.18,
                        1.05 - 0.79505,
                    ],
                    dtype=np.float32,
                )
            elif self._sweep_direction == Direction.LEFT:
                self._goal_pos = np.array(
                    [
                        pose_in_base.pose.position.x,
                        pose_in_base.pose.position.y + 0.18,
                        1.05 - 0.79505,
                    ],
                    dtype=np.float32,
                )
            else:
                self.get_logger().error(
                    f"Unknown sweep direction: {self._sweep_direction}"
                )
                return
            self.get_logger().info(f"Goal position set to: {self._goal_pos}")

        self._target_pos = np.array(
            [
                pose_in_base.pose.position.x,
                pose_in_base.pose.position.y,
                1.05 - 0.79505,
            ],
            dtype=np.float32,
        )

    def map_joint_angle(self, pos: float, index: int) -> float:
        """
        Map a simulation joint angle (in radians) to the real-world servo angle (in radians)

        Args:
            pos (float): Joint angle from simulation (in radians)
            index (int): Index of the joint

        Returns:
            float: Mapped joint angle withing the servo limits
        """
        L, U, inversed = self.SIM_DOF_ANGLE_LIMITS[index]
        A, B = self.SERVO_ANGLE_LIMITS[index]
        angle_deg = np.rad2deg(float(pos))
        # Check if the simulation angle is within limits
        if not L <= angle_deg <= U:
            self.get_logger().warn(
                f"Simulation joint {index} angle ({angle_deg}) out of range [{L}, {U}]. Clipping."
            )
            angle_deg = np.clip(angle_deg, L, U)
        # Map the angle from the simulation range to the servo range
        mapped = (angle_deg - L) * ((B - A) / (U - L)) + A
        if inversed:
            mapped = (B - A) - (mapped - A) + A
        # Verify the mapped angle is within servo limits
        if not A <= mapped <= B:
            raise Exception(
                f"Mapped joint {index} angle ({mapped}) out of servo range [{A}, {B}]."
            )
        return mapped

    def get_tcp_pose_in_base_link(self)-> np.ndarray:
        try:
            now = self.get_clock().now().to_msg()
            # base_link → tcp 변환 획득
            transform = self.tf_buffer.lookup_transform(
                target_frame="base_link",
                source_frame="tcp",
                time=rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )

            # 위치 텐서
            position_array = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                dtype=np.float32,
            )

            # 쿼터니언 텐서 (w, x, y, z 순서)
            orientation_array = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ],
                dtype=np.float32,
            )
            # 최종 Pose 텐서
            pose_array_in_base: np.ndarray = np.concatenate([position_array, orientation_array])

            return pose_array_in_base

        except Exception as e:
            self.get_logger().error(f"[TF Transform Error] {e}")

    def step(self):
        """
        Timer callback to compute and publish the next joint trajectory command.
        """
        r = self.create_rate(100.0)
        while rclpy.ok():
            if self._target_pos is None or self._goal_pos is None:
                self.get_logger().warn("Target position or goal position is not set.")
                continue

            # Set a constant target command for the robot (example values)
            self.current_pos = self.rtde_r.getActualQ()
            self.current_vel = self.rtde_r.getActualQd()
            self.robot.update_joint_state(self.current_pos, self.current_vel)
            moving_average = 0.8
            tcp_pose_in_base_link = self.get_tcp_pose_in_base_link()
            self.robot.update_tcp_state(tcp_pose_in_base_link)
            if tcp_pose_in_base_link is None:
                self.get_logger().warn("Current TCP pose is not available.")
                continue
            
            # Update Target Pose & Goal Pose
            self.robot.update_target_state(pos=self._target_pos)
            self.robot.update_goal_state(pos=self._goal_pos)
            self.robot.update_width(width=self._target_object_width)

            if not np.array_equal(self.robot.current_tcp_pose, np.zeros(7)):
                joint_pos = self.robot.forward(self.step_size)
                if joint_pos is not None:
                    if len(joint_pos) != 6:
                        raise Exception(
                            f"Expected 6 joint positions, got {len(joint_pos)}!"
                        )

                    joint_pos = np.array(joint_pos, dtype=np.float32)
                    cmd = [0] * 6

                    for i, pos in enumerate(joint_pos):
                        target_pos = self.map_joint_angle(pos, i)

                        cmd[i] = (
                            self.current_pos[i] * (1 - moving_average)
                            + target_pos * moving_average
                        )
                        
                    if self.current_pos is None or cmd is None:
                        return

                    # time start period
                    t_start = self.rtde_c.initPeriod()
                    self.rtde_c.servoJ(cmd, 0.1, 0.2, 1.0 / 100.0, 0.2, 300)

                    target_joint_state_msg = JointState()
                    target_joint_state_msg.header.stamp = self.get_clock().now().to_msg()
                    target_joint_state_msg.name = [
                        self.JOINT_NAMES[1],
                        self.JOINT_NAMES[0],
                        self.JOINT_NAMES[3],
                        self.JOINT_NAMES[4],
                        self.JOINT_NAMES[5],
                        self.JOINT_NAMES[2],
                    ]
                    target_joint_state_msg.position = [
                        cmd[1],
                        cmd[2],
                        cmd[3],
                        cmd[4],
                        cmd[5],
                        cmd[0],
                    ]

                    self.rtde_c.waitPeriod(t_start)

                self.i += 1
            
                tcp_data = PoseStamped()
                tcp_data.header.stamp = self.get_clock().now().to_msg()
                tcp_data.header.frame_id = "base_link"
                p = tcp_data.pose
                pos = p.position
                ori = p.orientation
                t = tcp_pose_in_base_link
                pos.x, pos.y, pos.z = float(t[0]), float(t[1]), float(t[2])
                ori.w, ori.x, ori.y, ori.z = float(t[3]), float(t[4]), float(t[5]), float(t[6])
                self._drl_logging_pub.publish(tcp_data)

                joint_vel_norm = np.linalg.norm(self.current_vel)
                if self.i > 200 and joint_vel_norm < 0.01:
                    self.get_logger().info("Robot has stopped moving. Ending episode.")
                    break
            
    def publish_pose_command_tf(self):
        if self.target_command is None:
            return

        pos: List[np.ndarray] = self.target_command[:3]
        quat: List[np.ndarray] = self.target_command[3:]  # [w, x, y, z]

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = "command"

        t.transform.translation.x = pos[0].item()
        t.transform.translation.y = pos[1].item()
        t.transform.translation.z = pos[2].item()

        t.transform.rotation.w = quat[0].item()
        t.transform.rotation.x = quat[1].item()
        t.transform.rotation.y = quat[2].item()
        t.transform.rotation.z = quat[3].item()

        self.tf_broadcaster.sendTransform(t)

    def reset(self):
        self.__start_time = datetime.datetime.now()
        self.rtde_c.moveJ(self.robot.default_pos[:6])
        self.rtde_c.stopJ()
        time.sleep(1)

    def stop(self):
        self.__end_time = datetime.datetime.now()
        self.rtde_c.stopJ()
        time.sleep(1)



def parse_args():
    """ROS2 인자를 제거한 뒤 argparse로 파싱"""
    filtered_args = remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser(description="Pose Estimate Node")
    parser.add_argument(
        "--log",
        type=bool,
        default=False,
        help="Set true for data logging."
    )
    parser.add_argument(
        "--episode_num",
        type=int,
        default=1,
        help="Episode number for logging purposes."
    )
    return parser.parse_args()  # [1:] to skip script name

def main(args=None):
    args = parse_args()
    rclpy.init(args=sys.argv)
    print(args.log)
    node = SweepPolicy(log=args.log, 
                       success_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_success_log.csv",
                       trajectory_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_trajectory_log0318_{args.episode_num}.csv",
                       object_log_file_path=f"/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/logs/drl/sweep_policy_object_log0318_{args.episode_num}.csv",
                       episode_num=args.episode_num
                       )
    
    import threading

    th = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    th.start()

    # --- 로깅 시작 서비스 요청 ---
    logger_client = node.create_client(LoggerRequest, 'logger_request')
    if logger_client.wait_for_service(timeout_sec=5.0):
        print("Logger service is available. Sending start request...")
        start_req = LoggerRequest.Request()
        start_req.start_logging = True
        start_req.direction = node._sweep_direction.value
        future = logger_client.call_async(start_req)
        while not future.done():
            time.sleep(0.05)
        node.get_logger().info("Logger start request sent.")
    else:
        node.get_logger().warn("Logger service not available, proceeding without logging service.")

    node.step()
    
    if logger_client.service_is_ready():
        stop_req = LoggerRequest.Request()
        stop_req.start_logging = False
        future = logger_client.call_async(stop_req)
        while not future.done():
            time.sleep(0.05)
        node.get_logger().info("Logger stop request sent.")
        
    th.join()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
