# Python Standard Library
import os
import sys
import time
import json
from abc import ABC, abstractmethod
from typing import List

# Third-Party Libraries
import numpy as np

# ROS2 Core Libraries
import rclpy
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_system_default
from rclpy.task import Future
from rclpy.time import Time

# ROS2 Message Types
from builtin_interfaces.msg import Duration as BuiltinDuration
from control_msgs.action import GripperCommand
from geometry_msgs.msg import *
from moveit_msgs.action import ExecuteTrajectory
from moveit_msgs.msg import *
from moveit_msgs.srv import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from shape_msgs.msg import *
from std_msgs.msg import *
from trajectory_msgs.msg import *
from visualization_msgs.msg import *
from custom_msgs.msg import *

# ROS2 TF Libraries
from tf2_ros import *

# Custom Base Package
from base_package.manager import Manager


class ForwardKinematics(object):
    @staticmethod
    def dh_transform(a, d, alpha, theta):
        """
        Denavit-Hartenberg 변환 행렬 생성 함수.
        :param a: 링크 길이
        :param d: 링크 오프셋
        :param alpha: 링크 간 회전
        :param theta: 조인트 각도
        :return: 4x4 변환 행렬
        """
        return np.array(
            [
                [
                    np.cos(theta),
                    -np.sin(theta) * np.cos(alpha),
                    np.sin(theta) * np.sin(alpha),
                    a * np.cos(theta),
                ],
                [
                    np.sin(theta),
                    np.cos(theta) * np.cos(alpha),
                    -np.cos(theta) * np.sin(alpha),
                    a * np.sin(theta),
                ],
                [0, np.sin(alpha), np.cos(alpha), d],
                [0, 0, 0, 1],
            ]
        )

    @staticmethod
    def forward_kinematics(joint_angles):
        """
        UR5e Forward Kinematics 계산 함수.
        :param joint_angles: 길이 6짜리 NumPy 배열, 조인트 각도 (라디안)
        :return: End Effector Pose (4x4 변환 행렬)
        """

        # UR5e DH 파라미터
        dh_params = [
            # (a_i, d_i, alpha_i, theta_i)
            (0, 0.1625, np.pi / 2, joint_angles[0]),  # Joint 1
            (-0.425, 0, 0, joint_angles[1]),  # Joint 2
            (-0.3922, 0, 0, joint_angles[2]),  # Joint 3
            (0, 0.1333, np.pi / 2, joint_angles[3]),  # Joint 4
            (0, 0.0997, -np.pi / 2, joint_angles[4]),  # Joint 5
            (0, 0.0996, 0, joint_angles[5]),  # Joint 6
        ]

        # 초기 변환 행렬 (Identity Matrix)
        t_matrix = np.eye(4)

        # X, Y축 뒤집기 행렬. 왜 이런지는 모르겠음.
        flip_transform = np.array(
            [[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # 각 DH 파라미터를 적용하여 누적 변환 계산
        for a, d, alpha, theta in dh_params:
            t_matrix = np.dot(
                t_matrix, ForwardKinematics.dh_transform(a, d, alpha, theta)
            )

        return np.dot(flip_transform, t_matrix)

    @staticmethod
    def parse_robot_trajectory_to_path(
        header: Header, joint_trajectory: RobotTrajectory
    ) -> Path:
        """
        shoulder_pan_joint,
        shoulder_lift_joint,
        elbow_joint,
        wrist_1_joint,
        wrist_2_joint,
        wrist_3_joint,
        해당 순서로 메세지를 변환합니다.
        """
        default_joint_order = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        path = Path()
        path.header = header

        joint_names = joint_trajectory.joint_trajectory.joint_names
        joint_order = [joint_names.index(joint) for joint in default_joint_order]

        position_keys = ["x", "y", "z"]
        orientation_keys = ["x", "y", "z", "w"]
        orientation = Quaternion(**dict(zip(orientation_keys, [0.0, 0.0, 0.0, 1.0])))

        for point in joint_trajectory.joint_trajectory.points:
            point: JointTrajectoryPoint

            joint_position = np.array(point.positions)
            joint_position = joint_position[joint_order]

            eef_pose = ForwardKinematics.forward_kinematics(joint_position)[
                :3, 3
            ]  # End Effector Pose, 길이 3 벡터

            eef_pose_stamped = PoseStamped(
                header=header,
                pose=Pose(
                    position=Point(**dict(zip(position_keys, eef_pose))),
                    orientation=orientation,
                ),
            )
            path.poses.append(eef_pose_stamped)

        return path


class GripperActionManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        # 액션 서버 이름은 launch와 환경에 맞게 수정
        self._action_client = ActionClient(
            self._node,
            GripperCommand,
            "/gripper/robotiq_gripper_controller/gripper_cmd",
        )
        self._is_finished = False
        self._is_success = False

    def control_gripper(self, open: bool = True):
        position = 0.0 if open else 0.8
        return self.send_goal(position)

    def send_goal(self, position: float, max_effort: float = 0.0):
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = position
        goal_msg.command.max_effort = max_effort

        self._action_client.wait_for_server()

        # >>> STEP 1. Send Goal
        future: Future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future: Future):
        # >>> STEP 2. Get Goal Handle. If goal is accepted, send result request
        goal_handle: ClientGoalHandle = future.result()

        if not goal_handle.accepted:
            # Reject된 경우
            self._is_finished = True
            self._is_success = False
            return None

        future = goal_handle.get_result_async()
        future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future: Future):
        self._is_finished = True
        self._is_success = True

    def feedback_callback(self, feedback_msg):
        # feedback은 optional, GripperCommand는 주로 result만 봄
        self._node.get_logger().info(f"Feedback: {feedback_msg}")

    @property
    def is_finished(self):
        temp = self._is_finished and self._is_success
        if temp:
            self._is_finished = False
            self._is_success = False
        return temp


class ServiceManager(Manager):
    def __init__(
        self, node: Node, service_name: str, service_type: type, *args, **kwargs
    ):
        super().__init__(node, *args, **kwargs)

        # >>> Parameters >>>
        self._error_code = {
            "NOT_INITIALIZED": 0,
            "SUCCESS": 1,
            "FAILURE": 99999,
            "PLANNING_FAILED": -1,
            "INVALID_MOTION_PLAN": -2,
            "MOTION_PLAN_INVALIDATED_BY_ENVIRONMENT_CHANGE": -3,
            "CONTROL_FAILED": -4,
            "UNABLE_TO_AQUIRE_SENSOR_DATA": -5,
            "TIMED_OUT": -6,
            "PREEMPTED": -7,
            "START_STATE_IN_COLLISION": -10,
            "START_STATE_VIOLATES_PATH_CONSTRAINTS": -11,
            "START_STATE_INVALID": -26,
            "GOAL_IN_COLLISION": -12,
            "GOAL_VIOLATES_PATH_CONSTRAINTS": -13,
            "GOAL_CONSTRAINTS_VIOLATED": -14,
            "GOAL_STATE_INVALID": -27,
            "UNRECOGNIZED_GOAL_TYPE": -28,
            "INVALID_GROUP_NAME": -15,
            "INVALID_GOAL_CONSTRAINTS": -16,
            "INVALID_ROBOT_STATE": -17,
            "INVALID_LINK_NAME": -18,
            "INVALID_OBJECT_NAME": -19,
            "FRAME_TRANSFORM_FAILURE": -21,
            "COLLISION_CHECKING_UNAVAILABLE": -22,
            "ROBOT_STATE_STALE": -23,
            "SENSOR_INFO_STALE": -24,
            "COMMUNICATION_FAILURE": -25,
            "CRASH": -29,
            "ABORT": -30,
            "NO_IK_SOLUTION": -31,
        }

        self._service_name = service_name
        self._service_type = service_type
        # <<< Parameters <<<

        # >>> Service Client >>>
        self._srv = self._node.create_client(service_type, service_name)

        while not self._srv.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().info(
                f"Service {service_name} not available, waiting again..."
            )
        # <<< Service Client <<<

    def get_error_code(self, code: int):
        for key, value in self._error_code.items():
            if value == code:
                return key

        return "UNKNOWN"

    def send_request(self, request):
        res = self._srv.call(request)
        return res

    @abstractmethod
    def run(self):
        """
        서비스 요청을 실행하는 메서드입니다.
        """
        raise NotImplementedError("run() must be implemented in the subclass")

    @abstractmethod
    def handle_response(self):
        """
        서비스 응답을 처리하는 메서드입니다.
        """
        raise NotImplementedError(
            "handle_response() must be implemented in the subclass"
        )


class FK_ServiceManager(ServiceManager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(
            node,
            service_name="/compute_fk",
            service_type=GetPositionFK,
            *args,
            **kwargs,
        )

    def run(
        self, joint_states: JointState, end_effector: str = "wrist_3_link"
    ) -> PoseStamped:
        if joint_states is None:
            raise ValueError("joint_states must be provided.")

        request = GetPositionFK.Request(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="base_link",
            ),
            fk_link_names=[end_effector],
            robot_state=(RobotState(joint_state=joint_states)),
        )

        response = self.send_request(request)
        result = self.handle_response(response)

        return result

    def handle_response(self, response: GetPositionFK.Response) -> PoseStamped:
        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.get_error_code(code)
            self._node.get_logger().warn(
                f"Error code in compute_fk service: {code}/{code_type}"
            )
            return None

        pose_stamped: PoseStamped = response.pose_stamped[0]
        return pose_stamped


class IK_ServiceManager(ServiceManager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(
            node,
            service_name="/compute_ik",
            service_type=GetPositionIK,
            *args,
            **kwargs,
        )

    def run(
        self,
        pose_stamped: PoseStamped,
        joint_states: JointState,
        end_effector: str = "wrist_3_link",
    ) -> RobotState:
        """
        :param pose_stamped: PoseStamped
            Desired end-effector pose
        :param joint_states: JointState
            Current joint states of the robot
        """
        if joint_states is None or pose_stamped is None:
            raise ValueError("joint_states and pose_stamped must be provided.")

        request = GetPositionIK.Request(
            ik_request=PositionIKRequest(
                group_name="ur_manipulator",
                robot_state=RobotState(joint_state=joint_states),
                avoid_collisions=False,
                ik_link_name=end_effector,
                pose_stamped=pose_stamped,
            )
        )

        response: GetPositionIK.Response = self.send_request(request)
        result = self.handle_response(response)

        return result

    def handle_response(self, response: GetPositionIK.Response) -> RobotState:
        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.get_error_code(code)
            self._node.get_logger().warn(
                f"Error code in compute_ik service: {code}/{code_type}"
            )
            return None

        solution: RobotState = response.solution
        return solution


class GetPlanningScene_ServiceManager(ServiceManager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(
            node,
            service_name="/get_planning_scene",
            service_type=GetPlanningScene,
            *args,
            **kwargs,
        )

    def run(self) -> PlanningScene:
        request = GetPlanningScene.Request()
        response: GetPlanningScene.Response = self.send_request(request)

        scene = self.handle_response(response)

        return scene

    def handle_response(self, response: GetPlanningScene.Response) -> PlanningScene:
        scene: PlanningScene = response.scene
        return scene


class ApplyPlanningScene_ServiceManager(ServiceManager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(
            node,
            service_name="/apply_planning_scene",
            service_type=ApplyPlanningScene,
            *args,
            **kwargs,
        )

    def run(self):
        pass

    def handle_response(self, response: ApplyPlanningScene.Response) -> bool:
        success = response.success
        return success

    def add_collistion_objects(
        self, collision_objects: List[CollisionObject], scene: PlanningScene
    ) -> bool:
        """
        MoveIt Planning Scene에 CollisionObject를 추가합니다.
        """
        if scene is None:
            raise ValueError("scene must be provided.")

        if collision_objects is None or len(collision_objects) == 0:
            raise ValueError("Collision objects must be provided.")

        if not isinstance(collision_objects[0], CollisionObject):
            raise ValueError(
                "collision_objects must be a list of CollisionObject objects."
            )

        scene.world.collision_objects = collision_objects

        request = ApplyPlanningScene.Request(scene=scene)
        response: ApplyPlanningScene.Response = self.send_request(request)

        result = self.handle_response(response)

        return result

    def reset_planning_scene(self, scene: PlanningScene) -> bool:
        """
        MoveIt Planning Scene을 초기화합니다. 모든 CollisionObject를 제거하고, 빈 Planning Scene을 생성합니다.
        """
        if scene is None:
            raise ValueError("scene must be provided.")

        # >>> STEP 1. Load the current planning scene >>>
        collision_objects = scene.world.collision_objects
        new_collision_objects = []

        # >>> STEP 2. Remove all collision objects >>>
        for obj in collision_objects:
            obj: CollisionObject

            new_obj = obj
            new_obj.operation = CollisionObject.REMOVE

            new_collision_objects.append(new_obj)

        scene.world.collision_objects = collision_objects

        # >>> STEP 3. Create a new planning scene >>>
        self._node.get_logger().info("Resetting the planning scene...")

        request = ApplyPlanningScene.Request(scene=scene)
        response: ApplyPlanningScene.Response = self.send_request(request)

        is_success_reset_wolrd = self.handle_response(response)

        return is_success_reset_wolrd

    def create_collision_object(
        self, id: str, header: Header, pose: Pose, scale: Vector3, operation
    ):
        """
        Primitive Type:
        - 0: CollisionObject.ADD
        - 1: CollisionObject.REMOVE
        - 2: CollisionObject.APPEND
        - 3: CollisionObject.MOVE
        """
        if isinstance(operation, int):
            operation = int.to_bytes(operation, byteorder="big")
        elif isinstance(operation, bytes):
            operation = operation
        else:
            raise TypeError("operation must be int or bytes")

        # 장애물 객체 생성
        return CollisionObject(
            id=id,
            header=header,
            operation=operation,
            primitives=[
                SolidPrimitive(
                    type=SolidPrimitive.BOX, dimensions=[scale.x, scale.y, scale.z]
                ),
            ],
            primitive_poses=[pose],
        )

    def collision_object_from_bbox_3d(
        self, header: Header, bbox_3d: BoundingBox3DMultiArray
    ) -> List[CollisionObject]:
        collision_objects = []

        for bbox in bbox_3d.data:
            bbox: BoundingBox3D

            collision_object = self.create_collision_object(
                id=bbox.cls,
                header=header,
                pose=bbox.pose,
                scale=bbox.scale,
                operation=CollisionObject.ADD,
            )
            collision_objects.append(collision_object)

        return collision_objects

    def append_default_collision_objects(
        self, scene: PlanningScene, header: Header
    ) -> bool:

        default_collision_objects_bbox = self.get_default_collision_objects()
        collision_objects = self.collision_object_from_bbox_3d(
            header=header,
            bbox_3d=default_collision_objects_bbox,
        )

        result: bool = self.add_collistion_objects(
            collision_objects=collision_objects, scene=scene
        )
        return result

    def get_default_collision_objects(self) -> List[BoundingBox3D]:
        data = []

        idx = 900

        data.append(
            BoundingBox3D(
                id=idx,
                cls="camera_box",
                pose=Pose(
                    position=Point(x=-0.04, y=-0.39, z=0.3),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.15, y=0.06, z=0.6),
            )
        )
        idx += 1

        # Add Plane Box
        data.append(
            BoundingBox3D(
                id=idx,
                cls="plane_box",
                pose=Pose(
                    position=Point(x=0.0, y=0.0, z=-0.5 - 0.01),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.8, y=0.44, z=1.0),
            )
        )
        idx += 1

        # Add Shelf Box
        data.append(
            BoundingBox3D(
                id=idx,
                cls="shelf_box1",
                pose=Pose(
                    position=Point(x=0.0, y=0.6, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                # scale=Vector3(x=0.8, y=0.44, z=10.0),
                scale=Vector3(x=0.8, y=0.44, z=0.48),  # 54
            )
        )
        idx += 1

        data.append(
            BoundingBox3D(
                id=idx,
                cls="shelf_box2",
                pose=Pose(
                    position=Point(x=0.0, y=0.6, z=0.7),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.8, y=0.44, z=0.04),
            )
        )
        idx += 1

        data.append(
            BoundingBox3D(
                id=idx,
                cls="shelf_side_box1",
                pose=Pose(
                    position=Point(x=0.54, y=0.6, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.1, y=0.4, z=1.0),
            )
        )
        idx += 1

        data.append(
            BoundingBox3D(
                id=idx,
                cls="shelf_side_box2",
                pose=Pose(
                    position=Point(x=-0.54, y=0.6, z=0.0),
                    orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
                scale=Vector3(x=0.1, y=0.4, z=1.0),
            )
        )
        idx += 1

        return BoundingBox3DMultiArray(data=data)


class CartesianPath_ServiceManager(ServiceManager):
    def __init__(self, node: Node, fraction_threshold: float = 0.999, *args, **kwargs):
        super().__init__(
            node,
            service_name="/compute_cartesian_path",
            service_type=GetCartesianPath,
            *args,
            **kwargs,
        )

        self._catesian_path_publisher = self._node.create_publisher(
            Path,
            self._node.get_name() + "/cartesian_path",
            qos_profile=qos_profile_system_default,
        )

        self._fraction_threshold = fraction_threshold

    def run(
        self,
        header: Header,
        waypoints: List[Pose],
        joint_states: JointState,
        end_effector: str = "wrist_3_link",
    ):
        # Exception handling
        if joint_states is None:
            raise ValueError("joint_states must be provided.")

        if len(waypoints) == 0:
            raise ValueError("waypoints must be provided.")

        if not isinstance(waypoints[0], Pose):
            raise ValueError("waypoints must be a list of Pose objects.")

        request = GetCartesianPath.Request(
            header=header,
            start_state=RobotState(joint_state=joint_states),
            group_name="ur_manipulator",
            link_name=end_effector,
            waypoints=waypoints,
            max_step=0.05,
            jump_threshold=5.0,
            avoid_collisions=True,
        )

        response: GetCartesianPath.Response = self.send_request(request)
        result = self.handle_response(response)

        return result

    def handle_response(
        self,
        response: GetCartesianPath.Response,
    ) -> RobotTrajectory:
        code = response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.get_error_code(code)
            self._node.get_logger().warn(
                f"Error code in compute_cartesian_path service: {code}/{code_type}"
            )
            return None

        trajectory: RobotTrajectory = response.solution
        fraction = response.fraction

        if fraction < self._fraction_threshold:
            self._node.get_logger().warn(
                f"Fraction is under {self._fraction_threshold}: {fraction}"
            )
            return None

        path: Path = ForwardKinematics.parse_robot_trajectory_to_path(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="world",
            ),
            joint_trajectory=trajectory,
        )
        for _ in range(10):
            self._catesian_path_publisher.publish(path)

        return trajectory


class KinematicPath_ServiceManager(ServiceManager):
    def __init__(
        self, node: Node, planning_group: str = "ur_manipulator", *args, **kwargs
    ):
        super().__init__(
            node,
            service_name="/plan_kinematic_path",
            service_type=GetMotionPlan,
            *args,
            **kwargs,
        )

        self._planning_group = planning_group
        self._kinematic_path_publisher = self._node.create_publisher(
            Path,
            self._node.get_name() + "/kinematic_path",
            qos_profile=qos_profile_system_default,
        )

    def run(
        self,
        goal_constraints: List[Constraints],
        path_constraints: Constraints,
        joint_states: JointState,
    ) -> RobotTrajectory:
        """
        :param goal_constraints: List[Constraints]
            List of goal constraints for the motion plan request.
        :param path_constraints: Constraints (Optional)
            Path constraints for the motion plan request.
        :param joint_states: JointState
            Current joint states of the robot.
        """
        # Exception handling
        if joint_states is None or goal_constraints is None:
            raise ValueError("joint_states and goal_constraints must be provided.")

        if not isinstance(goal_constraints[0], Constraints):
            raise ValueError("goal_constraints must be a list of Constraints objects.")

        # Unused Parameters:
        #     - workspace_parameters
        #     - path_constraints
        #     - trajectory_constraints
        #     - reference_trajectories
        #     - pipeline_id
        #     - planner_id

        request = GetMotionPlan.Request(
            motion_plan_request=MotionPlanRequest(
                start_state=RobotState(joint_state=joint_states),
                goal_constraints=goal_constraints,
                group_name=self._planning_group,
                num_planning_attempts=300,
                allowed_planning_time=10.0,
                max_velocity_scaling_factor=1.0,
                max_acceleration_scaling_factor=1.0,
            )
        )

        if path_constraints is not None:
            request.motion_plan_request.path_constraints = path_constraints

        response: GetMotionPlan.Response = self.send_request(request)
        trajectory: RobotTrajectory = self.handle_response(response)

        path: Path = ForwardKinematics.parse_robot_trajectory_to_path(
            header=Header(
                stamp=self._node.get_clock().now().to_msg(),
                frame_id="world",
            ),
            joint_trajectory=trajectory,
        )
        for _ in range(10):
            self._kinematic_path_publisher.publish(path)

        return trajectory

    def handle_response(
        self,
        response: GetMotionPlan.Response,
    ) -> RobotTrajectory:

        response_msg: MotionPlanResponse = response.motion_plan_response

        code = response_msg.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            code_type = self.get_error_code(code)
            self._node.get_logger().warn(
                f"Error code in plan_kinematic_path service: {code}/{code_type}"
            )
            return None

        trajectory: RobotTrajectory = response_msg.trajectory

        return trajectory

    def get_goal_constraint(
        self,
        goal_joint_states: JointState,
        tolerance: float = 0.05,
    ):
        """
        :param goal_joint_states: JointStatetamped
            Desired joint states of the robot.
        :param tolerance: float
            Tolerance for the joint constraints.
        :param end_effector: str
            End effector link name.
        """

        if goal_joint_states is None:
            raise ValueError("goal_joint_states must be provided.")

        name, position = goal_joint_states.name, goal_joint_states.position

        joint_constraints = []
        for n, p in zip(name, position):
            joint_constraint = JointConstraint(
                joint_name=n,
                position=p,
                weight=0.1,
                tolerance_above=tolerance,
                tolerance_below=tolerance,
            )
            joint_constraints.append(joint_constraint)

        constraints = Constraints(
            name="goal_constraints",
            joint_constraints=joint_constraints,
        )

        return constraints


class ExecuteTrajectory_ServiceManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(
            node,
            *args,
            **kwargs,
        )

        self._action_client = ActionClient(
            self._node, ExecuteTrajectory, "/execute_trajectory"
        )

    def run(self, trajectory: RobotTrajectory):
        goal_msg = ExecuteTrajectory.Goal(trajectory=trajectory)
        self._action_client.wait_for_server()

        response = self._action_client.send_goal(
            goal_msg, feedback_callback=self.feedback_callback
        )
        return response

    def feedback_callback(self, feedback_msg: ExecuteTrajectory.Feedback):
        # self._node.get_logger().info(f"Received feedback: {feedback_msg}")
        pass

    def scale_trajectory(self, trajectory: RobotTrajectory, scale_factor: float):
        def scale_duration(duration: BuiltinDuration, factor: float) -> BuiltinDuration:
            # 전체 시간을 나노초로 환산
            total_nanosec = duration.sec * 1_000_000_000 + duration.nanosec
            # n배
            scaled_nanosec = int(total_nanosec * factor)

            # 다시 sec과 nanosec으로 분리
            new_sec = scaled_nanosec // 1_000_000_000
            new_nanosec = scaled_nanosec % 1_000_000_000

            return BuiltinDuration(sec=new_sec, nanosec=new_nanosec)

        new_points = []
        for point in trajectory.joint_trajectory.points:
            point: JointTrajectoryPoint

            new_point = JointTrajectoryPoint(
                positions=np.array(point.positions),
                velocities=np.array(point.velocities) * scale_factor,
                accelerations=np.array(point.accelerations) * scale_factor,
                time_from_start=scale_duration(
                    duration=point.time_from_start, factor=(1.0 / scale_factor)
                ),
            )
            new_points.append(new_point)

        new_joint_trajectory = JointTrajectory(
            header=trajectory.joint_trajectory.header,
            joint_names=trajectory.joint_trajectory.joint_names,
            points=new_points,
        )

        new_trajectory = RobotTrajectory(
            joint_trajectory=new_joint_trajectory,
            multi_dof_joint_trajectory=trajectory.multi_dof_joint_trajectory,
        )

        return new_trajectory


class JointStatesManager(Manager):
    def __init__(self, node: Node, *args, **kwargs):
        super().__init__(node, *args, **kwargs)

        self._joint_states = None
        self._joint_states_subscriber = self._node.create_subscription(
            JointState,
            "/ur5e/joint_states",
            self.joint_states_callback,
            qos_profile=qos_profile_system_default,
        )

    def joint_states_callback(self, msg: JointState):
        self._joint_states = msg

    @property
    def joint_states(self):
        return self._joint_states


