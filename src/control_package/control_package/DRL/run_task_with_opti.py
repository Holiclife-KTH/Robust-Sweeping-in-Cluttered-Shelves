# ROS2
import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time

from rclpy.qos import QoSProfile, qos_profile_system_default
import tf2_ros

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

from control_package.DRL.ur import URSweepPolicy
from base_package.manager import TransformManager

# UR3
import rtde_control
import rtde_receive

# TF
from tf2_ros import *

# Python
import numpy as np

# import torch
import math
import time

# UR3
import rtde_control
import rtde_receive


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

    def __init__(self, fail_quietly: bool = False, verbose: bool = False):
        """Initialize the SweepPolicy node"""
        super().__init__("sweep_policy_node")

        self.robot = URSweepPolicy(
            model_path="/home/irol/workspace/project_th/src/ur5e_sweep/resource/260108"
        )

        # UR
        IP = "192.168.2.2"
        self.rtde_c = rtde_control.RTDEControlInterface(IP)
        self.rtde_r = rtde_receive.RTDEReceiveInterface(IP)

        self.target_command = np.zeros(7)
        self.step_size = 1.0 / 100.0  # 10 ms period = 100 Hz

        self.i = 0
        self.fail_quietly = fail_quietly
        self.verbose = verbose
        self.pub_freq = 100.0  # Hz
        self.current_pos = None  # Dictionary of current joint positions

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self._transform_manager = TransformManager(node=self)
        self._target_marker_id: int = 1
        self._marker_sub = self.create_subscription(
            MarkerArray,
            "/natnet_client_node/marker_array",
            self._marker_callback,
            qos_profile=qos_profile_system_default,
        )

        self._target_pos: np.ndarray = None
        self._goal_pos: np.ndarray = None

        # 여기에 타이머 추가
        # self.pose_command_tf_timer = self.create_timer(
        #     self.step_size,  # 10Hz 주기
        #     self.publish_pose_command_tf
        # )

        self._target_joint_pub = self.create_publisher(
            JointState,
            "/target_joint_states",
            qos_profile=qos_profile_system_default,
        )
        self._target_point_pub = self.create_publisher(
            PoseStamped,
            "/target_point",
            qos_profile=qos_profile_system_default,
        )

        self.min_traj_dur = 0  # Minimum trajectory duration in seconds
        self.timer = self.create_timer(self.step_size, self.step_callback)
        self.reset()

        self.get_logger().info("ReachPolicy node initialized.")

    def _marker_callback(self, msg: MarkerArray):
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
                    continue

                if isinstance(pose_in_base, PoseStamped):
                    self._target_point_pub.publish(pose_in_base)

                if self._target_pos is None:
                    self._goal_pos = np.array(
                        [
                            pose_in_base.pose.position.x,
                            pose_in_base.pose.position.y - 0.18,
                            1.05 - 0.79505,
                        ],
                        dtype=np.float32,
                    )
                    self.get_logger().info(f"Goal position set to: {self._goal_pos}")

                self._target_pos = np.array(
                    [
                        pose_in_base.pose.position.x,
                        pose_in_base.pose.position.y,
                        1.05 - 0.79505,
                    ],
                    dtype=np.float32,
                )

                break

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

    def get_tcp_pose_in_base_link(self):
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
            pose_tensor = np.concatenate([position_array, orientation_array])

            # self.get_logger().info(f"TCP Pose (Base Link): {pose_tensor[:3]}")

            self.robot.update_tcp_state(pose=pose_tensor)

        except Exception as e:
            self.get_logger().error(f"[TF Transform Error] {e}")

    def step_callback(self):
        """
        Timer callback to compute and publish the next joint trajectory command.
        """

        if self._target_pos is None or self._goal_pos is None:
            self.get_logger().warn("Target position or goal position is not set.")
            return

        print(f"Target Position: {self._target_pos}\nGoal Position: {self._goal_pos}\n")

        # Set a constant target command for the robot (example values)
        self.current_pos = self.rtde_r.getActualQ()
        self.current_vel = self.rtde_r.getActualQd()
        # self.get_logger().info(f"Joint pos: {self.current_pos}")
        self.robot.update_joint_state(self.current_pos, self.current_vel)
        moving_average = 0.95
        self.get_tcp_pose_in_base_link()
        # Update Target Pose & Goal Pose
        self.robot.update_target_state(pos=self._target_pos)
        self.robot.update_goal_state(pos=self._goal_pos)

        if not np.array_equal(self.robot.current_tcp_pose, np.zeros(7)):
            joint_pos = self.robot.forward(self.step_size)
            if joint_pos is not None:
                if len(joint_pos) != 6:
                    raise Exception(
                        f"Expected 6 joint positions, got {len(joint_pos)}!"
                    )

                joint_pos = np.array(joint_pos, dtype=np.float32)
                # offset = np.array([0.0, 0.01, 0.02, -0.01, 0.0, 0.0])
                # joint_pos += offset

                """
                joint_positions[1] += 0.01
                joint_positions[2] += 0.02
                joint_positions[3] -= 0.01
                
                """

                cmd = [0] * 6

                for i, pos in enumerate(joint_pos):
                    target_pos = self.map_joint_angle(pos, i)
                    cmd[i] = (
                        self.current_pos[i] * (1 - moving_average)
                        + target_pos * moving_average
                    )
                    # cmd[i] = target_pos
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

                self._target_joint_pub.publish(target_joint_state_msg)

                # self.rtde_c.moveJ(cmd, asynchronous=True)
                self.rtde_c.waitPeriod(t_start)
            #     # self.get_logger().info(f"current: {self.current_pos}")
            #     # self.get_logger().info(f"target: {joint_pos}")
            self.i += 1

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
        self.rtde_c.moveJ(self.robot.default_pos[:6])
        self.rtde_c.stopJ()
        time.sleep(1)

    def stop(self):
        self.rtde_c.stopJ()
        time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    node = SweepPolicy()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.robot.shutdown()
        node.stop()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
