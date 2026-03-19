from enum import Enum

import sys
import os
import json
import copy
import numpy as np
import datetime
import pandas as pd
import argparse
import time

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time
from rclpy.utilities import remove_ros_args

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from moveit_msgs.msg import *   
from builtin_interfaces.msg import Duration as BuiltinDuration
from custom_msgs.srv import LoggerRequest

# Custom
from moveit2_commander import FK_ServiceManager, ExecuteTrajectory_ServiceManager, CartesianPath_ServiceManager, KinematicPath_ServiceManager, IK_ServiceManager
from control_package.POE_Robot_Kinematics_Solver.robot_kinematic_solver import (
    RobotKinematicsPOE,
    transform_to_pose,
    pose_to_transform,
    transform_A_to_B,
    UR5E_CONFIG,
    IKResult,
)
from base_package.manager import TransformManager

class Direction(Enum):
    RIGHT = 0
    LEFT = 1

class PerceptionManager(object):
    def __init__(self, node: Node, target_marker_id: int = 1, sweep_direction: Direction = Direction.RIGHT):
        self.node = node
        self._target_marker_id = target_marker_id
        self._sweep_direction = sweep_direction

        self._sub = self.node.create_subscription(PoseStamped, '/pose_estimate/position', self.object_pose_callback, qos_profile_system_default)
        self._object_pose: PoseStamped = None

    def object_pose_callback(self, msg: PoseStamped):
        if msg is None:
            self.node.get_logger().warn(f"Any object pose received yet. Received: {msg}")
            return
        
        self._object_pose = msg

    def _ground_truth_object_position_callback(self, msg: MarkerArray):
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

                if self._gt_start_pos is None:
                    self._gt_start_pos = np.array(
                        [
                            pose_in_base.pose.position.x,
                            pose_in_base.pose.position.y,
                        ],
                        dtype=np.float32,
                    )

                    if self._sweep_direction == Direction.RIGHT:

                        self._gt_goal_pos = np.array(
                            [
                                pose_in_base.pose.position.x,
                                pose_in_base.pose.position.y - 0.18,
                                1.05 - 0.79505,
                            ],
                            dtype=np.float32,
                        )
                    elif self._sweep_direction == Direction.LEFT:
                        self._gt_goal_pos = np.array(
                            [
                                pose_in_base.pose.position.x,
                                pose_in_base.pose.position.y - 0.18,
                                1.05 - 0.79505,
                            ],
                            dtype=np.float32,   
                        )
                    else:
                        raise Exception(f"Unknown sweep direction: {self._sweep_direction}")

                self._gt_pos = np.array(
                    [
                        pose_in_base.pose.position.x,
                        pose_in_base.pose.position.y,
                    ],
                    dtype=np.float32,
                )

                break

    @property
    def object_pose(self)-> PoseStamped:
        return self._object_pose
    
    @property
    def object_pos_np(self) -> np.ndarray:
        '''
        Returns:
            np.ndarray: A 3D numpy array containing the x, y, z coordinates of the object pose.
        '''
        if self._object_pose is None:
            return None
        
        return np.array([
            self._object_pose.pose.position.x,
            self._object_pose.pose.position.y,
            self._object_pose.pose.position.z
        ])
    
    @property
    def gt_object_pos(self) -> np.ndarray:
        '''
        Returns:
            np.ndarray: A 3D numpy array containing the x, y, z coordinates of the ground truth object pose.
        '''
        if self._gt_pos is None:
            return None
        
        return self._gt_pos
    @property
    def gt_start_pos(self) -> np.ndarray:
        '''
        Returns:
            np.ndarray: A 3D numpy array containing the x, y, z coordinates of the ground truth start pose.
        '''
        if self._gt_start_pos is None:
            return None
        
        return self._gt_start_pos
    
    @property
    def gt_goal_pos(self) -> np.ndarray: 
        '''
        Returns:
            np.ndarray: A 3D numpy array containing the x, y, z coordinates of the ground truth goal pose.
        '''
        if self._gt_goal_pos is None:
            return None
        
        return self._gt_goal_pos

class JointStateManager(object):
    def __init__(self, node: Node):
        self.node = node

        self._sub = self.node.create_subscription(JointState, '/joint_states', self.joint_state_callback, qos_profile_system_default)
        self.__solver = RobotKinematicsPOE(UR5E_CONFIG)
        self._joint_state: JointState = None
        self._transform_manager = TransformManager(node)

        

    def joint_state_callback(self, msg: JointState):
        self._joint_state = msg


    

    @property
    def joint_state(self):
        return self._joint_state
    
    @property
    def tcp_pose(self):
        if self._joint_state is None:
            self.node.get_logger().warn("No joint state received yet.")
            return None
        
        joint_positions = self._joint_state.position
        np_joint_position = np.array([joint_positions[5], joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3], joint_positions[4]])
        tcp_transform = self.__solver.forward_kinematics(np_joint_position)
        np_tcp_pose = transform_to_pose(tcp_transform)

        tcp_pose: PoseStamped = Pose()
        tcp_pose.position.x = float(np_tcp_pose[0])
        tcp_pose.position.y = float(np_tcp_pose[1])
        tcp_pose.position.z = float(np_tcp_pose[2])

        tcp_pose_stamped_in_base_link = self._transform_manager.transform_pose(
            tcp_pose, target_frame="base_link", source_frame="base",
        )

        return tcp_pose_stamped_in_base_link

class LoggerManager(object):
    def __init__(self, node: Node, log: Bool=False, joint_manager:JointStateManager = None):
        self.node = node

        self.__joint_manager = joint_manager
        self.__log = log

        self.__tcp_publisher = self.node.create_publisher(
            PoseStamped,
            "/hitl_logging",
            qos_profile=qos_profile_system_default,
        )
        self.__hitl_tcp_logger = self.node.create_timer(0.01, self._hitl_tcp_log_callback)




    def _hitl_tcp_log_callback(self):
        if not self.__log and self.__joint_manager.tcp_pose is None:
            return

        self.__tcp_publisher.publish(self.__joint_manager.tcp_pose)

        

class HITLNode(Node):
    def __init__(self, log: bool = False):
        super().__init__('Human_in_the_Loop_node')

        self._direction = Direction.RIGHT
        self.fk_service_manager = FK_ServiceManager(self)
        # self.kinematic_path_service_manager = KinematicPath_ServiceManager(self, planning_group="ur5e_manipulator")
        self.traj_manager = CartesianPath_ServiceManager(self, planning_group="ur_manipulator", fraction_threshold=0.7)
        self.execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(self)
        self._joint_state_manager = JointStateManager(self)
        self._perception_manager = PerceptionManager(self, target_marker_id=1, sweep_direction=self._direction)
        self._logger_manager = LoggerManager(self, log=log, joint_manager=self._joint_state_manager)

        self.__initial_joint_state = [-2.0, 2.0, 0.0, 1.571, 1.571, 0.0] #[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan]

        # self.ik_service_manager = IK_ServiceManager(self, planning_group="ur5e_manipulator")

        self._gt_start_pos: np.ndarray = None
        self._gt_goal_pos: np.ndarray = None
        self._gt_pos: np.ndarray = None
        self._traj_tcp: pd.DataFrame = pd.DataFrame(columns=["tcp_x", "tcp_y", "tcp_z"])
        self._traj_obj: pd.DataFrame = pd.DataFrame(columns=["obj_x", "obj_y"])
        self._target_object_width: float = None



    def _hitl_log(self):
        if self._joint_state_manager.joint_state is None:
            return
        self.get_logger().info(f"Current TCP pose: {self.current_tcp_pose}, Current object pose: {self._perception_manager.object_pose}")


    

    def move(self):
        tcp_pose: PoseStamped = self.current_tcp_pose
        target_object_pos: np.ndarray = self._perception_manager.object_pos_np
        if target_object_pos is None:
            self.get_logger().warn("No object pose received yet.")
            return None
        
        if tcp_pose is None:
            return None
        
        if self._direction == Direction.RIGHT:
            waypoint_1 = Pose(
                position=Point(x=tcp_pose.pose.position.x, y=target_object_pos[1]+0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
            target_tcp_pose = Pose(
                position=Point(x=target_object_pos[0], y=target_object_pos[1]+0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        elif self._direction == Direction.LEFT:
            waypoint_1 = Pose(
                position=Point(x=tcp_pose.pose.position.x, y=target_object_pos[1]-0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
            target_tcp_pose = Pose(
                position=Point(x=target_object_pos[0], y=target_object_pos[1]-0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        else:
            self.get_logger().warn(f"Invalid direction: {self._direction}")
            return None

        self.get_logger().info(f"Target object position: {tcp_pose.pose.position}, Target TCP pose: {target_tcp_pose}")


        
        waypoints = [
            tcp_pose.pose,
            waypoint_1,
            target_tcp_pose
        ]


        traj: RobotTrajectory = self.traj_manager.run(header=Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='world'
        ), end_effector="gripper_link", joint_states=self._joint_state_manager.joint_state, waypoints=waypoints)

        if traj is None:
            self.get_logger().warn('Failed to compute Cartesian path.')
            return None

        new_traj = self.execute_trajectory_service_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.4
        )
        response = self.execute_trajectory_service_manager.run(trajectory=new_traj)
        
        # Step 2: Push
        current_tcp_pose = self.current_tcp_pose
        if self._direction == Direction.RIGHT:
            target_tcp_pose_push = Pose(
                position=Point(x=current_tcp_pose.pose.position.x, y=current_tcp_pose.pose.position.y-0.18, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        elif self._direction == Direction.LEFT:
            target_tcp_pose_push = Pose(
                position=Point(x=current_tcp_pose.pose.position.x, y=current_tcp_pose.pose.position.y+0.18, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5))
        else:
            self.get_logger().warn(f"Invalid direction: {self._direction}")
            return None
        waypoints_push = [current_tcp_pose.pose, target_tcp_pose_push]

        traj_push: RobotTrajectory = self.traj_manager.run(header=Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='world'
        ), end_effector="gripper_link", joint_states=self._joint_state_manager.joint_state, waypoints=waypoints_push)
        if traj_push is None:
            self.get_logger().warn('Failed to compute Cartesian path for push.')
            return None
        new_traj_push = self.execute_trajectory_service_manager.scale_trajectory(
            trajectory=traj_push,
            scale_factor=0.4
        )
        response_push = self.execute_trajectory_service_manager.run(trajectory=new_traj_push)

        return True
    
    def reset(self):
        if self._joint_state_manager.joint_state is None:
            self.get_logger().warn("No joint state received yet.")
            return None
        initial_joint_state:JointState = copy.deepcopy(self._joint_state_manager.joint_state)
        initial_joint_state.position = self.__initial_joint_state
        initial_tcp_pose: PoseStamped = self.fk(joint_states=initial_joint_state)
        self.get_logger().info(f"Initial TCP pose: {initial_tcp_pose}")
        if initial_tcp_pose is None:
            self.get_logger().warn("Failed to compute initial TCP pose.")
            return None
        waypoints = [
            initial_tcp_pose.pose
        ]
        traj: RobotTrajectory = self.traj_manager.run(header=Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='world'
        ), end_effector="gripper_link", joint_states=self._joint_state_manager.joint_state, waypoints=waypoints)    
        if traj is None:
            self.get_logger().warn('Failed to compute Cartesian path for reset.')
            return None
        new_traj = self.execute_trajectory_service_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.7
        )
        self.execute_trajectory_service_manager.run(trajectory=new_traj)

        return True


    def fk(self, joint_states: JointState = None) -> PoseStamped:
        eef_pose: PoseStamped = self.fk_service_manager.run(end_effector="gripper_link", joint_states=joint_states)
        return eef_pose
    
    @property
    def current_tcp_pose(self) -> PoseStamped:
        return self.fk(joint_states=self._joint_state_manager.joint_state)

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

    return parser.parse_args()

def main():
    import threading

    rclpy.init(args=None)
    args = parse_args()
    hitl_node = HITLNode(log=args.log)
    th = threading.Thread(target=rclpy.spin, args=(hitl_node,), daemon=True)
    th.start()

    r = hitl_node.create_rate(100.0)  # 1 Hz

    while hitl_node._joint_state_manager.joint_state is None:
        hitl_node.get_logger().info('Waiting for joint states...')
        r.sleep()
    
    hitl_node.reset()

    if args.log:
        # --- 로깅 시작 서비스 요청 ---
        logger_client = hitl_node.create_client(LoggerRequest, 'logger_request')
        if logger_client.wait_for_service(timeout_sec=5.0):
            print("Logger service is available. Sending start request...")
            start_req = LoggerRequest.Request()
            start_req.start_logging = True
            start_req.direction = hitl_node._direction.value
            future = logger_client.call_async(start_req)
            while not future.done():
                time.sleep(0.05)
            hitl_node.get_logger().info("Logger start request sent.")
        else:
            hitl_node.get_logger().warn("Logger service not available, proceeding without logging service.")    

    while rclpy.ok():

        if hitl_node.move():
            hitl_node.get_logger().info('Move successful!')
            break

        r.sleep()

    hitl_node.reset()

    if args.log and logger_client.service_is_ready():
        stop_req = LoggerRequest.Request()
        stop_req.start_logging = False
        future = logger_client.call_async(stop_req)
        while not future.done():
            time.sleep(0.05)
        hitl_node.get_logger().info("Logger stop request sent.")

    hitl_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()