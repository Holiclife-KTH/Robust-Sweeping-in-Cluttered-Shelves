# ROS2
import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, qos_profile_system_default

from tf2_ros import *

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

# from builtin_interfaces.msg import Duration
from rclpy.duration import Duration

# UR3
import rtde_control
import rtde_receive

# Python
import numpy as np
import enum
from rotutils import *
from scipy.spatial.transform import Rotation as R
from base_package.manager import TransformManager
from abc import ABC, abstractmethod
from typing import Type, Any

# import torch
import math
import time
from POE_Robot_Kinematics_Solver.robot_kinematic_solver import (
    RobotKinematicsPOE,
    transform_to_pose,
    pose_to_transform,
    transform_A_to_B,
    UR5E_CONFIG,
    IKResult,
)
import os
import sys
from loguru import logger


# --- 1. 하이퍼파라미터 설정 (Configuration) ---
CONFIG = {
    "prediction_horizon": 15,  # MPC 예측 구간 (L)
    "num_trajectories": 50,  # MPC 시뮬레이션 경로 수 (Q)
    "push_distance": 0.08,  # 원 위에서 Object 까지 밀어내는 거리
    "exact_push_distance": 0.004,  # 물체가 밀려나는 거리
    "virtual_radius": 0.1,  # 가상 원 반지름 (m)
    "init_data_points": 3,  # 초기 탐험 횟수
    "data_threshold_u": 0.1,  # Forward model 데이터 삭제 임계값
    "data_threshold_motion": 0.002,  # Inverse model 데이터 삭제 임계값
    "mpc_noise_scale": 0.1,  # MPC 시뮬레이션 시 노이즈 크기
    "lookahead_distance": 0.02,  # 경로 전방 주시 거리
    "position_weight": 10.0,  # 위치 비용 가중치
    "orientation_weight": 20.0,  # 각도 비용 가중치
    "progress_weight": 0.5,  # 진행 방향 비용 가중치
    "max_steps": 100,  # 최대 스텝 수
    "object_radius": 0.05,
    "alpha_range": [-0.5, 0.5],  # 푸시 위치 각도 범위 (rad)
    "beta_range": [-0.2, 0.2],  # 푸시 방향 각도 범위 (rad)
    "epsilon": 0.02,
}


# ── Pose Extractor (Strategy) ─────────────────────────────────
class PoseExtractor(ABC):
    """메시지에서 PoseStamped를 추출하는 전략 인터페이스"""

    @abstractmethod
    def extract(self, msg: Any, target_id: int, stamp) -> PoseStamped | None: ...


class MarkerPoseExtractor(PoseExtractor):
    """MarkerArray에서 특정 id의 Marker를 PoseStamped로 추출"""

    def extract(self, msg: MarkerArray, target_id: int, stamp) -> PoseStamped | None:
        for marker in msg.markers:
            if marker.id == target_id:
                return PoseStamped(
                    header=Header(
                        frame_id=marker.header.frame_id,
                        stamp=stamp,
                    ),
                    pose=marker.pose,
                )
        return None


class PoseStampedExtractor(PoseExtractor):
    """PoseStamped 메시지를 그대로 반환"""

    def extract(self, msg: PoseStamped, target_id: int, stamp) -> PoseStamped | None:
        return PoseStamped(
            header=Header(
                frame_id=msg.header.frame_id,
                stamp=stamp,
            ),
            pose=msg.pose,
        )


class MyLogger(object):
    def __init__(self, file: str):
        self.logger = logger

        is_existing = os.path.exists(file)

        self.logger.add(file, format="{message}", level="INFO")

        if not is_existing:
            self.logger.info("alpha,beta,motion.x,motion.y")

    def log(self, res: dict):
        alpha: float = res["input"]["alpha"]
        beta: float = res["input"]["beta"]
        motion: np.ndarray = res["output"]["motion"]

        self.logger.info(f"{alpha},{beta},{motion[0]},{motion[1]}")


# ── Object Manager ─────────────────────────────────────────────
class ObjectManager:
    def __init__(
        self,
        node: Node,
        id: int,
        base_frame: str = "base",
        topic_name: str = "/natnet_client_node/marker_array",
        message_type: Type = MarkerArray,
        extractor: PoseExtractor = None,
    ):
        self.__node = node
        self.__id = id
        self.__base_frame = base_frame
        self.__latest_pose: PoseStamped | None = None

        # 메시지 타입에 따라 기본 extractor 자동 선택
        if extractor is not None:
            self._extractor = extractor
        elif message_type == MarkerArray:
            self._extractor = MarkerPoseExtractor()
        elif message_type == PoseStamped:
            self._extractor = PoseStampedExtractor()
        else:
            raise ValueError(
                f"Unsupported message type: {message_type}. Provide a custom extractor."
            )

        self.__sub = self.__node.create_subscription(
            message_type,
            topic_name,
            self._callback,
            qos_profile=qos_profile_system_default,
        )

        self._transform_manager = TransformManager(node=self.__node)

    def _callback(self, msg):
        stamp = self.__node.get_clock().now().to_msg()
        pose = self._extractor.extract(msg, self.__id, stamp)
        if pose is not None:
            self.__latest_pose = pose

    # ── Properties ─────────────────────────────────────────────
    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, new_id: int):
        self.__id = new_id

    @property
    def base_frame(self) -> str:
        return self.__base_frame

    @base_frame.setter
    def base_frame(self, new_base_frame: str):
        self.__base_frame = new_base_frame

    @property
    def target_object(self) -> PoseStamped | None:
        if self.__latest_pose is None:
            return None

        pose_in_base: PoseStamped = self._transform_manager.transform_pose(
            pose=self.__latest_pose,
            target_frame=self.__base_frame,
            source_frame=self.__latest_pose.header.frame_id,
        )
        return pose_in_base

    @property
    def np_target_object_pose(self) -> np.ndarray | None:
        """Return: np.ndarray of shape (6,) -> [x, y, z, rx, ry, rz]"""
        target = self.target_object
        if target is None:
            return None

        r, p, y = euler_from_quaternion(
            [
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ]
        )

        return np.array(
            [
                target.pose.position.x,
                target.pose.position.y,
                target.pose.position.z,
                r,
                p,
                y,
            ]
        )


class UR5E_Controller(object):
    def __init__(self, node: Node):
        self.__node = node

        IP = "192.168.2.2"

        self.solver = RobotKinematicsPOE(UR5E_CONFIG)
        self.__rtde_c = rtde_control.RTDEControlInterface(IP)
        self.__rtde_r = rtde_receive.RTDEReceiveInterface(IP)

        self.__init_joint = [0.0, -2.0, 2.0, 0.0, 1.571, 1.571]

    @property
    def current_tcp_pose(self) -> np.ndarray:
        """
        Return: np.ndarray of shape (6,) -> [x, y, z, rx, ry, rz]
        """
        T = self.solver.forward_kinematics(self.__rtde_r.getActualQ())
        current_tcp_pose = transform_to_pose(T)
        return current_tcp_pose

    def moveL(self, pose: np.ndarray):
        # pose: [x, y, z, rx, ry, rz]
        pose_T = pose_to_transform(pose)
        q_current = self.__rtde_r.getActualQ()
        ik_result = self.solver.inverse_kinematics(pose_T, np.array(q_current))
        self.__rtde_c.moveJ(ik_result.joints, 1.0, 1.0)
        self.__rtde_c.stopJ()

    def moveTraj(
        self, pose_path: np.ndarray, velocity=0.5, acceleration=0.5, blend_radius=0.01
    ):
        total = len(pose_path)
        q_current = self.__rtde_r.getActualQ()
        joint_path = []

        for i, waypoint in enumerate(pose_path):
            pose = np.array(waypoint, dtype=np.float64)
            pose_T = pose_to_transform(pose)
            ik_result = self.solver.inverse_kinematics(pose_T, np.array(q_current))
            q_current = ik_result.joints  # 다음 IK의 seed로 사용

            br = 0.0 if i == total - 1 else blend_radius
            joint_path.append(list(ik_result.joints) + [velocity, acceleration, br])

        self.moveJointTraj(np.array(joint_path))

    def moveJointTraj(self, joint_path: np.ndarray):
        self.__rtde_c.moveJ(joint_path)
        self.__rtde_c.stopJ()

    def stop(self):
        self.__rtde_c.stopJ()

    def reset(self):
        self.__rtde_c.moveJ(self.__init_joint)
        self.__rtde_c.stopJ()
        self.__node.get_logger().info(
            f"Robot reset to initial position: {self.__init_joint}"
        )
        time.sleep(1.0)


class UnoPush(Node):
    def __init__(
        self, node_name: str = "data_collector_node", perception_mode: str = "opti"
    ):
        super().__init__(node_name)
        if perception_mode == "opti":
            self.get_logger().info("Running in OptiTrack mode.")
            self._object_manager = ObjectManager(
                node=self,
                id=1,
                base_frame="base",
                topic_name="/natnet_client_node/marker_array",
                message_type=MarkerArray,
            )
        elif perception_mode == "vision":
            self.get_logger().info("Running in Vision mode.")
            self._object_manager = ObjectManager(
                node=self,
                id=1,
                base_frame="base",
                topic_name="/pose_estimate/position",
                message_type=PoseStamped,
            )
        else:
            self.get_logger().warn(
                f"Unknown perception mode '{perception_mode}', defaulting to 'opti'."
            )
            self._object_manager = ObjectManager(node=self, id=1, base_frame="base")

        self._controller = UR5E_Controller(node=self)

        # System Variables
        self._collection_set_num = CONFIG["init_data_points"]
        self._object_virtual_radius = CONFIG["virtual_radius"]
        self._alpha_range = CONFIG["alpha_range"]
        self._beta_range = CONFIG["beta_range"]
        self._push_distance = CONFIG["push_distance"]

        # Temporal Variables
        self._last_alpha: float = None

    @property
    def alpha_range(self):
        return self._alpha_range

    @property
    def beta_range(self):
        return self._beta_range

    @property
    def set_num(self):
        return self._collection_set_num

    def stop(self):
        self._controller.stop()

    def _calculate_circle_path(
        self,
        prev_alpha: float,
        target_alpha: float,
        beta: float,
        obj_pos: PoseStamped,
        ignore: bool = False,
    ) -> np.ndarray:
        """
        Return: np.ndarray of shape (N, 6) -> [x, y, z, rx, ry, rz]
        """
        obj_pos: np.ndarray = np.array(
            [
                obj_pos.pose.position.x,
                obj_pos.pose.position.y,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

        if prev_alpha is None:
            prev_alpha = -np.pi / 2.0

        diff = target_alpha - prev_alpha
        steps = int(abs(math.degrees(diff)) // 5.0) + 1

        if abs(diff) < 1e-5:
            alphas = [target_alpha]
        elif diff > 0.0:
            alphas = np.linspace(prev_alpha, target_alpha, steps)
            alphas = np.sort(alphas)
        elif diff < 0.0:
            alphas = np.linspace(target_alpha, prev_alpha, steps)
            alphas = np.sort(alphas)[::-1]
        else:
            raise ValueError(
                "Invalid alpha values: prev_alpha={}, target_alpha={}".format(
                    prev_alpha, target_alpha
                )
            )

        offsets = np.array(
            [
                np.sin(alphas) * self._object_virtual_radius,
                np.cos(alphas) * self._object_virtual_radius,
                np.full_like(alphas, -1.0 * (1.05 - 0.79505 + 0.09)),
                np.zeros_like(alphas),
                np.full_like(alphas, 1.57),
                np.zeros_like(alphas),
            ]
        ).T

        path = obj_pos - offsets

        path[-1, 3] = (
            -target_alpha if abs(target_alpha) < 1.57 else np.pi - target_alpha
        ) + beta

        if ignore:
            path[-1, 3] = 0.0

        return path

    def _calculate_pushing_path(
        self,
        alpha: float,
        beta: float,
        obj_pos: PoseStamped,
    ) -> np.ndarray:
        """
        Return: np.ndarray of shape (6,) -> [x, y, z, rx, ry, rz]
        """

        current_tcp_pose = self._controller.current_tcp_pose

        np_target_object_pose = np.array(
            [
                obj_pos.pose.position.x,
                obj_pos.pose.position.y,
                obj_pos.pose.position.z,
                0.0,
                0.0,
                0.0,
            ]
        )

        direction = np_target_object_pose[:2] - current_tcp_pose[:2]

        if np.linalg.norm(direction) < 1e-6:
            return None

        direction_vector = direction / np.linalg.norm(direction)
        direction_vector = np.concatenate(
            (direction_vector, np.array([0.0, 0.0, 0.0, 0.0]))
        )

        # Apply beta rotation in XY plane
        direction_xy = direction_vector[:2]
        rotation_matrix = np.array(
            [[np.cos(beta), -np.sin(beta)], [np.sin(beta), np.cos(beta)]]
        )
        direction_vector[:2] = rotation_matrix @ direction_xy

        target_tcp_pose = current_tcp_pose + direction_vector * self._push_distance
        # target_tcp_pose[3] += beta

        return target_tcp_pose

    def run(self):
        raise NotImplementedError("This method should be implemented in the subclass.")


class DataCollector(UnoPush):
    def __init__(self, node_name: str = "data_collector_node"):
        super().__init__(node_name=node_name)
        # INIT
        self.__my_logger = MyLogger(file="collected_data8.csv")
        self._controller.reset()

    def run(self, alpha: float = 0.0, beta: float = 0.0):
        """
        현재 위치에서 Object Pose 방향으로 push.
        Return {
            "input": {
                "alpha": alpha : float
                "beta": beta: float,
            },
        "output": {
                "motion": motion : np.ndarray[float, float],  # Object Position 변화량 (XY)
            }
        """

        # STEP 1: 현재 Object Pose 에서, _object_virtual_radius 만큼 떨어진 위치로 이동 (왼쪽 / 오른쪽 -> alpha dependent)
        object_pose: PoseStamped = self._object_manager.target_object

        while object_pose is None:
            self.get_logger().warn("Waiting for object pose...")
            time.sleep(0.1)
            object_pose = self._object_manager.target_object

        # TODO: 힘수 이름 변경
        # STEP 2: 정해진 alpha, beta 만큼 각도 세팅 -> Radius 고려?
        path_1 = self._calculate_circle_path(
            prev_alpha=self._last_alpha,
            target_alpha=alpha,
            beta=beta,
            obj_pos=object_pose,
            ignore=False,
        )  # -> 원 형태로 주변에 alpha 각도로 세팅하는 path 생성 (P 지점)
        self._controller.moveTraj(pose_path=path_1)
        self._last_alpha = alpha

        # STEP 3: 현재 위치에서 Object Pose 방향으로 push (push distance: _push_distance)
        path_2 = self._calculate_pushing_path(alpha, beta, object_pose)

        # STEP 4: STEP 3 자세로 이동
        prev_object_pose = self._object_manager.np_target_object_pose[:2]
        self._controller.moveL(pose=path_2)
        current_object_pose = self._object_manager.np_target_object_pose[:2]

        # STEP 5: return
        res = {
            "input": {
                "alpha": alpha,
                "beta": beta,
            },
            "output": {
                "motion": current_object_pose - prev_object_pose,
            },
        }

        self.__my_logger.log(res)
        return res


def main(args=None):
    rclpy.init(args=args)
    data_collector = DataCollector()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(data_collector,))
    th.start()

    r = data_collector.create_rate(100.0)  # 10Hz
    try:

        for direction in [False, True]:
            alphas = np.linspace(
                data_collector.alpha_range[0],
                data_collector.alpha_range[1],
                # data_collector.set_num,
                10,
            ) - (np.pi if direction else 0.0)
            np.random.shuffle(alphas)

            for alpha in alphas:
                beta = np.random.uniform(
                    data_collector.beta_range[0], data_collector.beta_range[1]
                )
                res = data_collector.run(alpha=alpha, beta=beta)
                r.sleep()

        while rclpy.ok():
            r.sleep()

    except KeyboardInterrupt:
        data_collector.get_logger().info("Shutting down data collector...")
    except Exception as e:
        data_collector.get_logger().error(f"Error in main loop: {e}")
    finally:
        data_collector.stop()
        th.join()
        data_collector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
