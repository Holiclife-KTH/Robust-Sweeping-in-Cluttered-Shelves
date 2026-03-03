# Standard library
import math
import os
import sys
import time
import enum
from enum import Enum
import warnings

# Third-party: Data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R

# Third-party: ROS2
import rclpy
import rclpy.clock
from rclpy.node import Node
from rclpy.time import Time
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.duration import Duration
from tf2_ros import *
from tf2_geometry_msgs import do_transform_pose

# Third-party: ROS2 Messages
from std_msgs.msg import *
from geometry_msgs.msg import *
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import *
from nav_msgs.msg import *
from visualization_msgs.msg import *
from control_msgs.msg import JointTrajectoryControllerState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Local
from rotutils import *
from base_package.manager import TransformManager
from POE_Robot_Kinematics_Solver.robot_kinematic_solver import (
    RobotKinematicsPOE,
    transform_to_pose,
    pose_to_transform,
    transform_A_to_B,
    UR5E_CONFIG,
    IKResult,
)
from loguru import logger

# From custom package
from uno_push.data_collertor import (
    UnoPush,
    CONFIG,
)


# 경고 메시지 숨기기 (GPR 수렴 경고 등)
warnings.filterwarnings("ignore")


class Direction(Enum):
    RIGHT = 0
    LEFT = 1


# --- 3. GPR 역학 모델 (Learner) with Normalization ---
class DynamicsModel:
    """Gaussian Process Regression 기반 역학 모델.
    Forward model: u -> motion (행동 → 결과 움직임)
    Inverse model: motion -> u (원하는 움직임 → 필요한 행동)
    """

    def __init__(self, data_threshold):
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(
            noise_level=0.01
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=2, alpha=1e-6
        )
        self.X_train = []
        self.Y_train = []
        self.is_fitted = False
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.data_threshold = data_threshold

    def update(self, x, y):
        """새로운 데이터 포인트 (x, y)를 추가하고 GPR 재학습."""
        x = np.array(x).flatten()
        y = np.array(y).flatten()

        # 데이터 관리 (너무 가까운 데이터는 교체)
        if len(self.X_train) > 0:
            dists = np.linalg.norm(np.array(self.X_train) - x, axis=1)
            if np.min(dists) < self.data_threshold:
                idx = np.argmin(dists)
                del self.X_train[idx]
                del self.Y_train[idx]

        self.X_train.append(x.tolist())
        self.Y_train.append(y.tolist())

        # 충분한 데이터가 있을 때만 학습
        if len(self.X_train) >= 5:
            X = np.array(self.X_train)
            Y = np.array(self.Y_train)

            # 정규화
            X_scaled = self.x_scaler.fit_transform(X)
            Y_scaled = self.y_scaler.fit_transform(Y)

            self.gpr.fit(X_scaled, Y_scaled)
            self.is_fitted = True

    def predict(self, x):
        """입력 x에 대한 예측값 반환."""
        if not self.is_fitted:
            return np.zeros(3)

        x = np.atleast_2d(x)
        x_scaled = self.x_scaler.transform(x)
        y_scaled = self.gpr.predict(x_scaled)
        y = self.y_scaler.inverse_transform(y_scaled.reshape(1, -1))
        return y.flatten()

    def predict_with_std(self, x):
        """예측값과 불확실성(표준편차) 반환."""
        if not self.is_fitted:
            return np.zeros(3), np.ones(3)

        x = np.atleast_2d(x)
        x_scaled = self.x_scaler.transform(x)
        y_scaled, std_scaled = self.gpr.predict(x_scaled, return_std=True)
        y = self.y_scaler.inverse_transform(y_scaled.reshape(1, -1))
        return y.flatten(), std_scaled


# --- 4. MPC 제어기 (Controller) ---
class MPCController:
    """Model Predictive Control 기반 푸시 행동 결정.
    Forward model과 Inverse model을 활용하여
    예측 구간 내 최저 비용 행동을 탐색한다.
    """

    def __init__(self, f_model: DynamicsModel, i_model: DynamicsModel):
        self.f_model: DynamicsModel = f_model  # Forward model: u -> motion
        self.i_model: DynamicsModel = i_model  # Inverse model: motion -> u
        self.best_prev_action = None

    def find_target_on_path(self, current_pose, ref_path):
        """경로 상에서 lookahead 거리에 있는 목표점 찾기."""
        dists = np.linalg.norm(ref_path[:, :2] - current_pose[:2], axis=1)
        nearest_idx = np.argmin(dists)

        # for i in range(nearest_idx, len(ref_path)):
        #     y_path = ref_path[i, 1]
        #     y_current = current_pose[1]
        #     if y_path > y_current:
        #         nearest_idx = i
        #         break
        #     nearest_idx += 1

        # 골 근처에서는 lookahead 거리를 줄임
        dist_to_goal = np.linalg.norm(current_pose[:2] - ref_path[-1, :2])
        # if dist_to_goal < 0.15:
        #     lookahead = CONFIG["lookahead_distance"] * 0.3
        # else:
        lookahead = CONFIG["lookahead_distance"]

        accumulated_dist = 0
        target_idx = nearest_idx

        # Ensure we search forward along the path from nearest point
        for i in range(nearest_idx, len(ref_path) - 1):
            segment_dist = np.linalg.norm(ref_path[i + 1, :2] - ref_path[i, :2])
            accumulated_dist += segment_dist

            if accumulated_dist >= lookahead:
                target_idx = i + 1
                break
        else:
            target_idx = len(ref_path) - 1

        return ref_path[target_idx], nearest_idx

    def compute_cost(
        self,
        sim_pose: np.ndarray,
        target_pose: np.ndarray,
        ref_path: np.ndarray,
        path_idx,
    ):
        """비용 함수 계산."""
        goal_pose = ref_path[-1, :2]

        # 1. 위치 오차
        pos_error = np.linalg.norm(sim_pose[:2] - target_pose[:2])
        x_error = abs(sim_pose[0] - target_pose[0])
        y_error = abs(sim_pose[1] - target_pose[1])

        # 3. 경로 진행 방향 보상
        progress = 0
        if path_idx < len(ref_path) - 1:
            path_direction = ref_path[path_idx + 1, :2] - ref_path[path_idx, :2]
            path_direction = path_direction / (np.linalg.norm(path_direction) + 1e-6)
            move_direction = sim_pose - ref_path[path_idx, :2]
            move_dist = np.linalg.norm(move_direction)
            if move_dist > 1e-6:
                move_direction = move_direction / move_dist
                progress = np.dot(path_direction, move_direction)

        cost = (
            CONFIG["position_weight"] * x_error * 4.0
            + CONFIG["position_weight"] * y_error
            # + CONFIG["orientation_weight"] * abs(heading_error)
            # - CONFIG["progress_weight"] * progress
        )
        return cost

    def clip_action(self, u):
        """행동을 유효 범위로 제한."""
        if abs(u[0]) < np.pi / 2.0:
            # 오른쪽으로 미는 경우
            alpha = np.clip(u[0], CONFIG["alpha_range"][0], CONFIG["alpha_range"][1])
        else:
            alpha = np.clip(
                u[0], CONFIG["alpha_range"][0] - np.pi, CONFIG["alpha_range"][1] - np.pi
            )

        beta = np.clip(u[1], CONFIG["beta_range"][0], CONFIG["beta_range"][1])
        return np.array([alpha, beta])

    def get_action(self, current_pose: np.ndarray, ref_path: np.ndarray):
        """MPC를 통해 최적 행동 결정.
        Q개의 랜덤 궤적을 시뮬레이션하여 가장 낮은 비용의
        첫 번째 행동을 반환한다.
        return: [alpha, beta] 형태의 행동 벡터
        """
        best_cost = float("inf")
        best_action = None

        target_pose, path_idx = self.find_target_on_path(current_pose, ref_path)
        sim_pose = current_pose.copy()

        # 목표 움직임 계산 (물체 좌표계 기준)
        dx_body = target_pose[0] - sim_pose[0]
        dy_body = target_pose[1] - sim_pose[1]

        # 방향 유지, 크기를 push_distance로 정규화
        distance_body = np.linalg.norm([dx_body, dy_body])
        if distance_body > 1e-6:
            scale = 0.02 / distance_body
            dx_body *= scale
            dy_body *= scale

        desired_motion = np.array([dx_body, dy_body])
        # print("목표 움직임:", desired_motion, np.linalg.norm(desired_motion))

        for traj_idx in range(CONFIG["num_trajectories"]):
            sim_pose = current_pose.copy()
            traj_cost = 0
            first_action = None

            for k in range(CONFIG["prediction_horizon"]):
                # 현재 시뮬레이션 위치에서 목표점 재계산
                target_pose_k, path_idx_k = self.find_target_on_path(sim_pose, ref_path)

                # 목표 움직임 계산 (물체 좌표계 기준)
                dx_body = target_pose_k[0] - sim_pose[0]
                dy_body = target_pose_k[1] - sim_pose[1]

                # 방향 유지, 크기를 push_distance로 정규화
                distance_body = np.linalg.norm([dx_body, dy_body])
                if distance_body > 1e-6:
                    scale = CONFIG["exact_push_distance"] / distance_body
                    dx_body *= scale
                    dy_body *= scale

                desired_motion = np.array([dx_body, dy_body])

                # 노이즈 추가 (탐색용, 첫 궤적은 exploitation)
                if traj_idx > 0:
                    noise_scale = CONFIG["mpc_noise_scale"] * (
                        1.0 - k / CONFIG["prediction_horizon"]
                    )
                    noise = np.random.normal(0, noise_scale, 2)
                    desired_motion = desired_motion + noise

                # 역모델로 행동(u) 예측
                u_pred = self.i_model.predict(desired_motion)
                u_pred = self.clip_action(u_pred)
                if k == 0:
                    first_action = u_pred

                # 순방향 모델로 결과 예측
                pred_motion = self.f_model.predict(u_pred)

                # 시뮬레이션 상태 업데이트
                sim_pose += pred_motion

                # 비용 계산 (할인 적용)
                discount = 0.95**k
                step_cost = self.compute_cost(
                    sim_pose, target_pose_k, ref_path, path_idx_k
                )
                traj_cost += discount * step_cost

            if traj_cost < best_cost:
                best_cost = traj_cost
                best_action = first_action

        # 안전장치
        if best_action is None:
            self.get_logger().warn(
                "MPC가 유효한 행동을 찾지 못했습니다. 기본 행동으로 대체합니다."
            )
            best_action = np.array([0.0, 0.0])

        best_pred_motion = self.f_model.predict(best_action)
        # self.get_logger().info(
        #     f"선택된 행동: {best_action}, 예측 움직임: {best_pred_motion}, 움직임 크기: {np.linalg.norm(best_pred_motion)}"
        # )

        self.best_prev_action = best_action
        return best_action


def data_post_processing(log_file_path, direction: Direction):
    """
    수집된 로그 파일에서 행동과 결과 데이터를 추출하여 모델 학습에 적합한 형태로 변환.
    Return:
    - params: (N, 2) 형태의 행동 데이터 배열 (alpha, beta)
    - motions: (N, 2) 형태의 결과 움직임 데이터 배열 (dx, dy)
    """
    global CONFIG

    log_file_df = pd.read_csv(log_file_path)

    alphas = log_file_df["alpha"].to_numpy()
    betas = log_file_df["beta"].to_numpy()
    motion_xs = log_file_df["motion.x"].to_numpy()
    motion_ys = log_file_df["motion.y"].to_numpy()

    params = np.column_stack((alphas, betas))
    motions = np.column_stack((motion_xs, motion_ys))

    # MASK1
    distance_threshold = 0.005
    masks = np.linalg.norm(motions, axis=1) > distance_threshold

    motions = motions[masks]
    params = params[masks]

    # MASK2
    # range_1 = (-0.5, 0.5)
    # range_2 = (-0.5 - np.pi, 0.5 - np.pi)

    # masks = ((params[:, 0] >= range_1[0]) & (params[:, 0] <= range_1[1])) | (
    #     (params[:, 0] >= range_2[0]) & (params[:, 0] <= range_2[1])
    # )
    # motions = motions[masks]
    # params = params[masks]

    alpha_threshold = np.pi / 2.0
    masks = (
        (np.abs(params[:, 0]) < alpha_threshold)
        if direction == Direction.RIGHT
        else (np.abs(params[:, 0]) > alpha_threshold)
    )
    motions = motions[masks]
    params = params[masks]

    # Update push_distance based on the mean motion distance in the dataset
    # mean_motion_distance = np.mean(np.linalg.norm(motions, axis=1))
    # CONFIG["push_distance"] = np.clip(mean_motion_distance, 0.005, 0.1)

    normalized_motions = (
        motions / np.linalg.norm(motions, axis=1, keepdims=True)
    ) * 0.02

    return params, normalized_motions


class TrajectoryPublisher(object):
    def __init__(self, node: Node, topic_name: str = "trajectory", base_frame="world"):
        self._node = node
        self._base_frame = base_frame

        self._publisher = self._node.create_publisher(
            Path,
            self._node.get_name() + "/" + topic_name,
            qos_profile=qos_profile_system_default,
        )

    def _parse_trajectory(self, trajectory: np.ndarray) -> Path:
        path_msg = Path(
            header=Header(
                frame_id=self._base_frame,
                stamp=self._node.get_clock().now().to_msg(),
            ),
            poses=[],
        )

        for point in trajectory:
            point: np.ndarray  # [x, y, z, r, p, y]
            pose = PoseStamped(
                header=Header(
                    frame_id=self._base_frame,
                    stamp=self._node.get_clock().now().to_msg(),
                ),
                pose=Pose(
                    position=Point(
                        x=point[0],
                        y=point[1],
                        z=0.0,
                    ),
                    orientation=Quaternion(
                        x=0.0,
                        y=0.0,
                        z=0.0,
                        w=1.0,
                    ),
                ),
            )
            path_msg.poses.append(pose)

        return path_msg

    def publish_trajectory(self, trajectory: Path | np.ndarray):
        if isinstance(trajectory, np.ndarray):
            trajectory = self._parse_trajectory(trajectory)
        self._publisher.publish(trajectory)


class UnoMPC(UnoPush):
    def __init__(self):
        super().__init__(node_name="uno_mpc_node")

        # System Parameters
        self.__hz = 20.0  # 제어 주파수
        self.__distance_threshold = 0.01  # 목표 도달로 간주하는 거리 (m)
        self.__log: bool = True  # 로그 출력 여부

        self.__f_model = DynamicsModel(data_threshold=CONFIG["data_threshold_u"])
        self.__i_model = DynamicsModel(data_threshold=CONFIG["data_threshold_motion"])
        self.__mpc = MPCController(self.__f_model, self.__i_model)

        self._direction = Direction.LEFT  # 푸시 방향 설정 (RIGHT 또는 LEFT)
        self.__prev_u = None

        self.__params, self.__motions = data_post_processing(
            "/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/control_package/resource/collected_data5.csv",
            direction=self._direction,
        )

        self._trajectory_publisher = TrajectoryPublisher(
            self, topic_name="mpc_trajectory", base_frame="base"
        )

        # INIT
        # self._push_distance = CONFIG["exact_push_distance"]
        self._controller.reset()

    def reset(self):
        self._controller.reset()
        self.__prev_u = None
        self._last_alpha = None

    def _calculate_target_path(
        self, direction: Direction, ref_path_length=50
    ) -> np.ndarray:
        """
        Return: [x, y, z, r, p, y] 형태의 참조 경로 배열 (6 * N)
        Warning: Orientation과 Z축 처리 할 것!!
        """
        object_pose = self._object_manager.np_target_object_pose  # [x, y, z, r, p, y]
        offset = np.array([0.0, 0.18, 0.0, 0.0, 0.0, 0.0])

        target_pose = object_pose + (
            offset * (1.0 if direction == Direction.RIGHT else -1.0)
        )

        reference_path = np.linspace(object_pose, target_pose, num=ref_path_length)
        return reference_path

    def _calculate_sine_target_path(
        self, direction: Direction, ref_path_length=50
    ) -> np.ndarray:
        """
        Return: [x, y, z, r, p, y] 형태의 참조 경로 배열 (6 * N)
        Warning: Orientation과 Z축 처리 할 것!!
        """
        object_pose = self._object_manager.np_target_object_pose  # [x, y, z, r, p, y]
        offset = np.array([0.0, 0.25, 0.0, 0.0, 0.0, 0.0])
        target_pose = object_pose + (
            offset * (1.0 if direction == Direction.RIGHT else -1.0)
        )

        # Create a sine wave curve from object_pose to target_pose
        t = np.linspace(0, 1, ref_path_length)

        # Sine wave amplitude and frequency
        amplitude = 0.05
        frequency = np.pi

        # Generate sine wave in the perpendicular direction
        sine_offset = amplitude * np.sin(frequency * t)

        # Interpolate linearly from object_pose to target_pose
        linear_path = np.array(
            [object_pose + s * (target_pose - object_pose) for s in t]
        )

        # Add sine wave perpendicular to the main direction
        direction = target_pose - object_pose
        direction_norm = direction / (np.linalg.norm(direction) + 1e-6)
        # Perpendicular direction (rotate 90 degrees in xy plane)
        perp_direction = np.array([-direction_norm[1], direction_norm[0], 0, 0, 0, 0])

        reference_path = linear_path + np.outer(sine_offset, perp_direction)

        return np.array(reference_path)

    def _push(self, alpha: float, beta: float, obj_pos: PoseStamped):

        # STEP 1: 원형 위치 세팅
        path_1 = self._calculate_circle_path(
            prev_alpha=self._last_alpha,
            target_alpha=alpha,
            beta=beta,
            obj_pos=obj_pos,
            ignore=False,
        )
        if self._last_alpha is None:
            current_tcp_pose = self._controller.current_tcp_pose
            pre_path = np.linspace(current_tcp_pose, path_1[0], num=20)
            self._controller.moveTraj(pose_path=pre_path)

        p_prime = self.smoothened_execute(
            virtual_circle_path=path_1, alpha_t=alpha, beta_t=beta
        )

        if p_prime is not None:
            self._controller.moveL(pose=p_prime)
        else:
            self._controller.moveTraj(pose_path=path_1)
        self._last_alpha = alpha

        # STEP 2: 원형 위치에서 푸시
        path_2 = self._calculate_pushing_path(alpha, beta, obj_pos)

        # STEP 4: STEP 3 자세로 이동
        prev_object_pose = self._object_manager.np_target_object_pose[:2]
        self._controller.moveL(pose=path_2)
        current_object_pose = self._object_manager.np_target_object_pose[:2]

        motion_real = current_object_pose - prev_object_pose

        return motion_real

    def smoothened_execute(
        self, virtual_circle_path: list, alpha_t: float, beta_t: float
    ):
        """
        논문의 Algorithm 5: SmoothenedExecute 구현

        Args:
            current_action: [alpha, beta] (현재 스텝에서 MPC가 생성한 제어 입력)
            object_pose: 물체의 현재 위치 [x, y]
            ee_pose: 로봇 엔드이펙터(그리퍼)의 현재 위치 [x, y]
            R: 가상 원의 반경 (Virtual Circle Radius, 물체 크기보다 크게 설정)
            d: 푸시 거리 (Push distance)
            epsilon: 스무딩을 적용할 이전 행동과의 차이 임계값 (sigma)
        """

        last_tcp_pose = virtual_circle_path[0]
        target_tcp_pose = virtual_circle_path[
            -1
        ].copy()  # 원형 경로의 마지막 지점이 푸시 시작 위치가 됨
        ee_diff = np.linalg.norm(last_tcp_pose[:2] - target_tcp_pose[:2])

        if ee_diff < CONFIG["epsilon"]:
            # --- Smoothened Execution 적용 ---

            # 3~5. 물체와 엔드이펙터 위치 및 거리 계산 (Algorithm 5: line 3-5)
            p_t = self._object_manager.np_target_object_pose[:2]
            p_EE = self._controller.current_tcp_pose[:2]
            r = np.linalg.norm(p_t - p_EE) + 0.002
            # 거리가 0이 되는 것을 방지 (안전 장치)
            r = max(r, 1e-6)

            # 7. 보조 각도 gamma_t 계산 (Algorithm 5: line 7 - Law of Sines)
            # 수학적 안정성을 위해 arcsin 내부 값을 [-1, 1]로 클리핑
            sin_val = np.clip(
                (CONFIG["virtual_radius"] * np.sin(beta_t)) / r, -1.0, 1.0
            )
            gamma_t = alpha_t + beta_t - np.arcsin(sin_val)

            # 8. P' 위치 계산 (Algorithm 5: line 8)
            # 사용자가 orientation을 무시하므로, 물체 중심을 원점으로 하는 글로벌 각도로 바로 치환 가능
            p_prime_offset = np.array([r * np.sin(gamma_t), r * np.cos(gamma_t)])
            p_prime = target_tcp_pose
            p_prime[:2] = p_t - p_prime_offset
            p_prime[3] = (
                -alpha_t + beta_t if abs(alpha_t) < 1.57 else np.pi - alpha_t + beta_t
            )

            # 9~10. P'로 이동 후 푸시 (Algorithm 5: line 9-10)
            # 푸시 방향은 alpha_t와 beta_t가 결정하는 목표 진입 각도의 반대 방향(물체를 향함)
            push_angle = alpha_t + beta_t + np.pi

            self.get_logger().info(
                f"Smoothened Execution 발동: 궤적을 부드럽게 연결합니다. P'={p_prime}"
            )

            # [TODO] 컨트롤러에 p_prime 위치로 이동 후 push_dir 방향으로 d 만큼 밀도록 명령 전달
            # self._controller.moveL(pose=p_prime)
            # self._controller.push(direction=push_dir, distance=d)

            return p_prime

        else:
            return None

    def waiting(self):
        # STEP 2: 정해진 alpha, beta 만큼 각도 세팅 -> Radius 고려?
        path_1 = self._calculate_circle_path(
            prev_alpha=self._last_alpha,
            target_alpha=-np.pi / 2.0,
            beta=0.0,
            obj_pos=self._object_manager.target_object,
            ignore=True,
        )  # -> 원 형태로 주변에 alpha 각도로 세팅하는 path 생성 (P 지점)

        self._last_alpha = -np.pi / 2.0
        self._controller.moveTraj(pose_path=path_1)

    def run(self):
        r = self.create_rate(100.0)
        while rclpy.ok() and self._object_manager.target_object is None:
            r.sleep()

        # STEP 1: 모델 업데이트
        for k, (param, motion) in enumerate(zip(self.__params, self.__motions)):
            self.__f_model.update(param, motion)
            self.__i_model.update(motion, param)

            if self.__log:
                print(
                    f"Data Point {k+1}/{len(self.__params)}: Param={param}, Motion={motion}"
                )

        print(f"   Forward model data points: {len(self.__f_model.X_train)}")
        print(f"   Inverse model data points: {len(self.__i_model.X_train)}")

        # STEP 2: MPC 제어 루프
        reference_path: np.ndarray = self._calculate_target_path(
            self._direction, ref_path_length=500
        )
        # reference_path = self._calculate_sine_target_path(
        #     Direction.RIGHT, ref_path_length=50
        # )

        self.get_logger().info(f"목표 위치: {reference_path[-1, :2]}")

        r = self.create_rate(self.__hz)
        while rclpy.ok():

            self._trajectory_publisher.publish_trajectory(reference_path)
            self.get_logger().info(
                f"현재 위치: {self._object_manager.np_target_object_pose[:2]}"
            )

            current_position_xy = self._object_manager.np_target_object_pose[
                :2
            ]  # [x, y, z, r, p, y]

            last_distance = np.linalg.norm(current_position_xy - reference_path[-1, :2])
            if last_distance < self.__distance_threshold:
                self.get_logger().info(
                    f"목표에 도달했습니다!: 거리={last_distance:.3f}m"
                )
                self.get_logger().info(f"목표 위치: {reference_path[-1, :2]}")
                self.get_logger().info(f"최종 위치: {current_position_xy}")
                break

            u_opt = self.__mpc.get_action(current_position_xy, reference_path)

            # TODO: 실제 행동 실행 코드 추가 (예: self.__controller.push(u_opt))
            real_motion = self._push(
                alpha=u_opt[0],
                beta=u_opt[1],
                obj_pos=self._object_manager.target_object,
            )

            self.__prev_u = u_opt
            skip = False
            if np.linalg.norm(real_motion) < (CONFIG["exact_push_distance"] * 0.9):
                self.get_logger().warn(
                    "실제 움직임이 너무 작습니다. Push distance를 늘립니다."
                )
                self._push_distance += (
                    np.abs(np.linalg.norm(real_motion) - CONFIG["exact_push_distance"])
                    * 0.8
                )
                skip = True

            elif np.linalg.norm(real_motion) > (CONFIG["exact_push_distance"] * 1.1):
                self.get_logger().warn(
                    "실제 움직임이 너무 큽니다. Push distance를 줄입니다."
                )
                self._push_distance = np.clip(
                    self._push_distance
                    - (
                        np.abs(
                            np.linalg.norm(real_motion) - CONFIG["exact_push_distance"]
                        )
                        * 0.8
                    ),
                    0.0,
                    0.1,
                )

            self.get_logger().info(
                f"실제 움직인 거리: {np.linalg.norm(real_motion) * 1000.0:.3f}mm, Push distance: {self._push_distance * 1000.0:.3f}mm"
            )

            normalized_real_motion = (
                real_motion / (np.linalg.norm(real_motion) + 1e-6)
            ) * 0.02

            if not skip:
                self.__f_model.update(u_opt, normalized_real_motion)
                self.__i_model.update(normalized_real_motion, u_opt)

            r.sleep()


def main(args=None):
    rclpy.init(args=args)
    uno_mpc_node = UnoMPC()

    import threading

    th = threading.Thread(target=rclpy.spin, args=(uno_mpc_node,), daemon=True)
    th.start()

    uno_mpc_node.run()

    r = uno_mpc_node.create_rate(30.0)

    for _ in range(5):  # 종료 전에 잠시 대기하여 로그 출력이 완료되도록 함
        uno_mpc_node.waiting()
        r.sleep()

    uno_mpc_node.reset()
    r.sleep()

    while rclpy.ok():
        r.sleep()

    th.join()
    uno_mpc_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
