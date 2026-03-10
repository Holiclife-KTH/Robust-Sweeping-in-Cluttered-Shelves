from enum import Enum

import sys
import os
import json
import copy
import numpy as np

# ROS2
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.time import Time

# ROS2 Messages
from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *
from moveit_msgs.msg import *   
from builtin_interfaces.msg import Duration as BuiltinDuration

# Custom
from moveit2_commander import FK_ServiceManager, ExecuteTrajectory_ServiceManager, CartesianPath_ServiceManager, KinematicPath_ServiceManager, IK_ServiceManager

class Direction(Enum):
    RIGHT = 0
    LEFT = 1

class PerceptionManager(object):
    def __init__(self, node: Node):
        self.node = node

        self._sub = self.node.create_subscription(PoseStamped, '/pose_estimate/position', self.object_pose_callback, qos_profile_system_default)
        self._object_pose: PoseStamped = None

    def object_pose_callback(self, msg: PoseStamped):
        if msg is None:
            self.node.get_logger().warn(f"Any object pose received yet. Received: {msg}")
            return
        
        self._object_pose = msg

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

class JointStateManager(object):
    def __init__(self, node: Node):
        self.node = node

        self._sub = self.node.create_subscription(JointState, '/joint_states', self.joint_state_callback, qos_profile_system_default)
        self._joint_state: JointState = None

    def joint_state_callback(self, msg: JointState):
        self._joint_state = msg

    @property
    def joint_state(self):
        return self._joint_state


class HITLNode(Node):
    def __init__(self):
        super().__init__('Human_in_the_Loop_node')

        self.fk_service_manager = FK_ServiceManager(self)
        # self.kinematic_path_service_manager = KinematicPath_ServiceManager(self, planning_group="ur5e_manipulator")
        self.traj_manager = CartesianPath_ServiceManager(self, planning_group="ur_manipulator", fraction_threshold=0.7)
        self.execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(self)
        self._joint_state_manager = JointStateManager(self)
        self._perception_manager = PerceptionManager(self)

        self.__direction = Direction.RIGHT
        self.__initial_joint_state = [-2.0, 2.0, 0.0, 1.571, 1.571, 0.0] #[shoulder_lift, elbow, wrist_1, wrist_2, wrist_3, shoulder_pan]

        # self.ik_service_manager = IK_ServiceManager(self, planning_group="ur5e_manipulator")

    def move(self):
        tcp_pose: PoseStamped = self.current_tcp_pose
        target_object_pos: np.ndarray = self._perception_manager.object_pos_np
        if target_object_pos is None:
            self.get_logger().warn("No object pose received yet.")
            return None
        
        if tcp_pose is None:
            return None
        
        if self.__direction == Direction.RIGHT:
            target_tcp_pose = Pose(
                position=Point(x=target_object_pos[0], y=target_object_pos[1]+0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        elif self.__direction == Direction.LEFT:
            target_tcp_pose = Pose(
                position=Point(x=target_object_pos[0], y=target_object_pos[1]-0.1, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        else:
            self.get_logger().warn(f"Invalid direction: {self.__direction}")
            return None

        self.get_logger().info(f"Target object position: {tcp_pose.pose.position}, Target TCP pose: {target_tcp_pose}")

        waypoints = [
            tcp_pose.pose,
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
        if self.__direction == Direction.RIGHT:
            target_tcp_pose_push = Pose(
                position=Point(x=current_tcp_pose.pose.position.x, y=current_tcp_pose.pose.position.y-0.18, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5)
            )
        elif self.__direction == Direction.LEFT:
            target_tcp_pose_push = Pose(
                position=Point(x=current_tcp_pose.pose.position.x, y=current_tcp_pose.pose.position.y+0.18, z=0.32),
                orientation=Quaternion(x=0.5, y=0.5, z=0.5, w=0.5))
        else:
            self.get_logger().warn(f"Invalid direction: {self.__direction}")
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


def main(args=None):
    import threading

    rclpy.init(args=args)

    hitl_node = HITLNode()
    th = threading.Thread(target=rclpy.spin, args=(hitl_node,), daemon=True)
    th.start()

    r = hitl_node.create_rate(50.0)  # 1 Hz

    while hitl_node._joint_state_manager.joint_state is None:
        hitl_node.get_logger().info('Waiting for joint states...')
        r.sleep()
    hitl_node.reset()
    while rclpy.ok():

        if hitl_node.move():
            hitl_node.get_logger().info('Move successful!')
            break

        r.sleep()

    hitl_node.reset()

    hitl_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()