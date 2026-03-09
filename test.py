import sys
import os
import json

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


class TestNode(Node):
    def __init__(self):
        super().__init__('test_node')

        self.fk_service_manager = FK_ServiceManager(self)
        # self.kinematic_path_service_manager = KinematicPath_ServiceManager(self, planning_group="ur5e_manipulator")
        self.traj_manager = CartesianPath_ServiceManager(self, planning_group="ur_manipulator", fraction_threshold=0.7)
        self.execute_trajectory_service_manager = ExecuteTrajectory_ServiceManager(self)
        self.joint_state_manager = JointStateManager(self)
        # self.ik_service_manager = IK_ServiceManager(self, planning_group="ur5e_manipulator")

    def move(self):
        eef_pose: PoseStamped = self.fk()
        if eef_pose is None:
            return None
        
        target_eef_pose = Pose(
            position=Point(x=eef_pose.pose.position.x, y=eef_pose.pose.position.y, z=eef_pose.pose.position.z + 0.03),
            orientation=eef_pose.pose.orientation
        )

        waypoints = [
            Pose(
                position=Point(x=eef_pose.pose.position.x, y=eef_pose.pose.position.y, z=eef_pose.pose.position.z + 0.03),
                orientation=eef_pose.pose.orientation
            ),
            Pose(
                position=Point(x=eef_pose.pose.position.x, y=eef_pose.pose.position.y, z=eef_pose.pose.position.z),
                orientation=eef_pose.pose.orientation
            ),
            Pose(
                position=Point(x=eef_pose.pose.position.x, y=eef_pose.pose.position.y, z=eef_pose.pose.position.z + 0.03),
                orientation=eef_pose.pose.orientation
            ),
            Pose(
                position=Point(x=eef_pose.pose.position.x, y=eef_pose.pose.position.y, z=eef_pose.pose.position.z),
                orientation=eef_pose.pose.orientation
            )
        ]

        print(f"Waypoints: {eef_pose.pose} -> {waypoints}")

        # traj: RobotTrajectory = self.kinematic_path_service_manager.run(
        #     goal_constraints=self.ik_service_manager.run(end_effector="wrist_3_link", target_pose=target_eef_pose)

        # )

        # self.kinematic_path_service_manager.get_goal_constraint()

        traj: RobotTrajectory = self.traj_manager.run(header=Header(
            stamp=self.get_clock().now().to_msg(),
            frame_id='world'
        ), end_effector="wrist_3_link", joint_states=self.joint_state_manager.joint_state, waypoints=waypoints)

        if traj is None:
            self.get_logger().warn('Failed to compute Cartesian path.')
            return None

        new_traj = self.execute_trajectory_service_manager.scale_trajectory(
            trajectory=traj,
            scale_factor=0.2
        )

        self.execute_trajectory_service_manager.run(trajectory=new_traj)

        return True


    def fk(self):
        current_joint_state = self.joint_state_manager.joint_state
        if current_joint_state is None:
            self.get_logger().info('Waiting for joint states...')
            return None
        
        eef_pose: PoseStamped = self.fk_service_manager.run(end_effector="wrist_3_link", joint_states=current_joint_state)
        return eef_pose


def main(args=None):
    import threading

    rclpy.init(args=args)

    test_node = TestNode()
    th = threading.Thread(target=rclpy.spin, args=(test_node,), daemon=True)
    th.start()

    r = test_node.create_rate(30.0)  # 1 Hz


    while rclpy.ok():

        if test_node.move():
            test_node.get_logger().info('Move successful!')
            break

        r.sleep()

    test_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()