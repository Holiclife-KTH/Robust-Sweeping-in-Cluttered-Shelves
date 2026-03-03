import numpy as np
import torch
from control_package.DRL.policy_controller import PolicyController
import os
import sys

from loguru import logger


class URSweepPolicy(PolicyController):
    """Policy controller for UR Sweep using a pre-trained policy model"""

    def __init__(self, model_path: str) -> None:
        """Initialize the URSweepPolicy instance."""
        super().__init__()

        assert os.path.exists(model_path), "Model path does not exist"

        self.dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        # Load the pre-trained policy model and environment configuration
        # YOU NEED TO  CHANGE THE PATH

        self.load_policy(
            f"{model_path}/exported/policy.pt",
            f"{model_path}/params/env.yaml",
        )

        self._action_scale = 0.5
        self._previous_action = np.zeros(7)
        self.action = np.zeros(7)
        self._policy_counter = 0
        self.has_joint_data = False
        self.has_tcp_data = False
        self.current_joint_positions = np.zeros(8, dtype=np.float32)
        self.current_joint_velocities = np.zeros(8, dtype=np.float32)
        self.current_tcp_pose = np.zeros(7, dtype=np.float32)

        self.target_pos: np.ndarray = None
        self.goal_pos: np.ndarray = None

    def update_joint_state(self, position, velocity) -> None:
        """
        Update the current joint state.
        Args:
            position (_type_): A list or array of joint positions
            velocity (_type_): A list or array of joint velocities
        """

        self.current_joint_positions = np.array(
            position[: self.num_joints], dtype=np.float32
        )
        self.current_joint_velocities = np.array(
            velocity[: self.num_joints], dtype=np.float32
        )
        self.has_joint_data = True

    def update_tcp_state(self, pose) -> None:
        """
        Update the current tcp state.

        Args:
            pose (_type_): A list or array of tcp point
        """
        self.current_tcp_pose = np.array(pose)
        self.has_tcp_data = True

    def update_target_state(self, pos: np.ndarray) -> None:
        assert isinstance(pos, np.ndarray), "Position must be a numpy array."
        assert pos.shape == (3,), "Position must be a 3-element array."

        print(f"Update Target Position: {pos}")

        if pos[0] < 0.5:
            self.target_pos = self.target_pos
        else:
            self.target_pos = pos

    def update_goal_state(self, pos: np.ndarray) -> None:
        assert isinstance(pos, np.ndarray), "Goal position must be a numpy array."
        assert pos.shape == (3,), "Goal position must be a 3-element array."

        self.goal_pos = pos

    def _compute_observation(self) -> np.ndarray:
        """
        Compute the observation vector for the policy network.

        Args:
            command (np.ndarray): The target command vector

        Returns:
            np.ndarray: An observation vector if joint data is available, otherwise None.
        """

        if not (self.has_joint_data or self.has_tcp_data):
            print("No joint or TCP data available.")
            return None

        if self.target_pos is None or self.goal_pos is None:
            print("Target or goal position is None.")
            return None

        obs = np.zeros(35)
        obs[:6] = self.current_joint_positions - self.default_pos[:6]

        if self._previous_action[-1] < 0:
            obs[6:8] = 0.4
        else:
            obs[6:8] = 0.04

        obs[8:14] = self.current_joint_velocities
        obs[14:21] = self._previous_action
        obs[21:24] = self.target_pos
        obs[24] = 0.1  # self._target_object_width
        obs[25:32] = self.current_tcp_pose
        obs[32:35] = self.goal_pos  # self._goal_pos

        obs = np.expand_dims(obs, axis=0).astype(np.float32)

        return obs

    def forward(self, dt: float) -> np.ndarray:
        """
        Compute the next joint positions based on the policy

        Args:
            dt (float): Time step for the forward pass.
            command (np.ndarray): The target command vector.

        Returns:
            np.ndarray: The computed joint positions if joint data is available.
        """

        if not (self.has_joint_data or self.has_tcp_data):
            return None

        obs = None

        if self._policy_counter % self._decimation == 0:

            obs = self._compute_observation()
            if obs is None:
                return None
            self._previous_action = self.action.copy()
            self.action = self._compute_action(obs)

        processed_action = self.action * self._action_scale
        joint_positions = processed_action[:6] + self.default_pos[:6]

        self._policy_counter += 1

        return joint_positions

    def shutdown(self):
        pass
