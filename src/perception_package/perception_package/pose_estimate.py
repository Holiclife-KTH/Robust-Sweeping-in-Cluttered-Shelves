import cv2
import numpy as np
import argparse
import sys

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.duration import Duration
from rclpy.utilities import remove_ros_args

from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PointStamped, PoseStamped, Pose
from custom_msgs.msg import BBox, BBoxMultiArray
from tf2_geometry_msgs import do_transform_pose
from tf2_ros import Buffer, TransformListener, TransformBroadcaster, TransformStamped


# ── Configuration ──────────────────────────────────────────────
POINTCLOUD_TOPIC = "/helios/pointcloud_rgb"
BBOX_TOPIC = "/segmentation/bboxes"
POSITION_TOPIC = "/pose_estimate/position"
CALIBRATION_FILE = "/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/ROS2_Helios2_RGB_KIT/src/lucid_camera_node/resource/orientation.yml"
# ───────────────────────────────────────────────────────────────


class PoseEstimateNode(Node):
    def __init__(self, target_cls: str):
        super().__init__("pose_estimate_node")
        self._target_cls = target_cls
        self.get_logger().info(f"Target class: {self._target_cls}")

        # ── State ──────────────────────────────────────────────
        self._points_xyz: np.ndarray | None = None
        self._projected_2d: np.ndarray | None = None

        # Calibration (lazy-loaded)
        self._camera_matrix = None
        self._dist_coeffs = None
        self._rvec = None
        self._tvec = None

        # Cached transform
        self._cached_transform = None
        self._transform_update_timer = self.create_timer(0.1, self._update_transform)

        # ── Subscribers ────────────────────────────────────────
        self.create_subscription(
            PointCloud2,
            POINTCLOUD_TOPIC,
            self._pointcloud_callback,
            qos_profile_system_default,
        )
        self.create_subscription(
            BBoxMultiArray,
            BBOX_TOPIC,
            self._bbox_callback,
            qos_profile_system_default,
        )

        # ── Publisher ──────────────────────────────────────────
        self._position_pub = self.create_publisher(
            PoseStamped,
            POSITION_TOPIC,
            qos_profile_system_default,
        )

        self.__tf_buffer = Buffer(
            node=self, cache_time=Duration(seconds=0, nanoseconds=500000000)
        )
        self.__transform_listener = TransformListener(
            buffer=self.__tf_buffer, node=self, qos=qos_profile_system_default
        )
        self.__transform_broadcaster = TransformBroadcaster(
            node=self, qos=qos_profile_system_default
        )

        self.get_logger().info(f"Target : {self._target_cls}")
        self.get_logger().info(f"Sub PC : {POINTCLOUD_TOPIC}")
        self.get_logger().info(f"Sub BB : {BBOX_TOPIC}")
        self.get_logger().info(f"Pub pos: {POSITION_TOPIC}")

    # ── Cached TF Update ───────────────────────────────────────
    def _update_transform(self):
        try:
            self._cached_transform = self.__tf_buffer.lookup_transform(
                "base_link", "helios_camera", rclpy.time.Time()
            )
        except Exception:
            pass  # keep previous cache

    # ── Calibration ────────────────────────────────────────────
    def _load_calibration(self):
        fs = cv2.FileStorage(CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            self.get_logger().error(f"Cannot open calibration file: {CALIBRATION_FILE}")
            raise RuntimeError("Calibration file not found")

        self._camera_matrix = fs.getNode("cameraMatrix").mat()
        self._dist_coeffs = fs.getNode("distCoeffs").mat()
        self._rvec = fs.getNode("rotationVector").mat()
        self._tvec = fs.getNode("translationVector").mat()
        fs.release()

        self.get_logger().info("Calibration loaded")

    # ── PointCloud Callback ────────────────────────────────────
    def _pointcloud_callback(self, msg: PointCloud2):
        points = self._pointcloud2_to_numpy(msg)
        if points is None or len(points) == 0:
            return

        self._points_xyz = points
        self._projected_2d = self._project_3d_to_2d(points)

    # ── BBox Callback ──────────────────────────────────────────
    def _bbox_callback(self, msg: BBoxMultiArray):
        if self._points_xyz is None or self._projected_2d is None:
            return

        # Find target bbox (highest confidence)
        target_bboxes = [b for b in msg.data if b.cls == self._target_cls]
        if not target_bboxes:
            return

        target = max(target_bboxes, key=lambda b: b.conf)

        # Extract 3D points inside the bbox
        roi_points = self._extract_roi_points(target)
        if roi_points is None or len(roi_points) < 3:
            self.get_logger().warn(f"[{self._target_cls}] Not enough points in bbox")
            return

        # Remove outliers & compute centroid
        clean = self._remove_outliers_iqr(roi_points)
        centroid = np.median(clean, axis=0)

        # Publish position and log
        self._target_object_position_publisher(centroid)

        self.get_logger().info(
            f"[{self._target_cls}] pos: ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})"
        )

    def _target_object_position_publisher(self, position: np.ndarray):
        if self._cached_transform is None:
            self.get_logger().warn("No cached transform available yet")
            return

        pose_camera = Pose()
        pose_camera.position.x = float(position[0])
        pose_camera.position.y = float(position[1])
        pose_camera.position.z = float(position[2])
        # identity quaternion
        pose_camera.orientation.w = 1.0

        pose_in_base = do_transform_pose(pose_camera, self._cached_transform)

        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "base_link"
        pose_stamped.pose.position = pose_in_base.position

        self._position_pub.publish(pose_stamped)

    # ── Core Logic ─────────────────────────────────────────────
    def _extract_roi_points(self, bbox: BBox) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox.bbox
        if x2 <= x1 or y2 <= y1:
            return None

        mask = (
            (self._projected_2d[:, 0] >= x1)
            & (self._projected_2d[:, 0] <= x2)
            & (self._projected_2d[:, 1] >= y1)
            & (self._projected_2d[:, 1] <= y2)
        )
        roi = self._points_xyz[mask]
        if len(roi) == 0:
            return None

        # Filter NaN and unreasonable depth
        valid = ~np.isnan(roi).any(axis=1) & (roi[:, 2] > 0.05) & (roi[:, 2] < 3.0)
        result = roi[valid]
        return result if len(result) >= 3 else None

    def _project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray | None:
        if self._camera_matrix is None:
            self._load_calibration()

        points_mm = points_3d * 1000.0  # m → mm (lucid_node convention)

        projected, _ = cv2.projectPoints(
            points_mm.astype(np.float64),
            self._rvec,
            self._tvec,
            self._camera_matrix,
            self._dist_coeffs,
        )
        return projected.reshape(-1, 2)

    # ── PointCloud2 Conversion ─────────────────────────────────
    @staticmethod
    def _pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray | None:
        offsets = {f.name: f.offset for f in msg.fields}
        if not all(k in offsets for k in ("x", "y", "z")):
            return None

        n = msg.width * msg.height
        step = msg.point_step
        raw = np.frombuffer(msg.data, dtype=np.uint8)

        if len(raw) < n * step:
            return None

        data = raw[: n * step].reshape(n, step)
        xyz = np.column_stack(
            [
                data[:, offsets["x"] : offsets["x"] + 4]
                .copy()
                .view(np.float32)
                .flatten(),
                data[:, offsets["y"] : offsets["y"] + 4]
                .copy()
                .view(np.float32)
                .flatten(),
                data[:, offsets["z"] : offsets["z"] + 4]
                .copy()
                .view(np.float32)
                .flatten(),
            ]
        )
        return xyz

    # ── Outlier Removal ────────────────────────────────────────
    @staticmethod
    def _remove_outliers_iqr(points: np.ndarray) -> np.ndarray:
        z = points[:, 2]
        q1, q3 = np.percentile(z, [25, 75])
        iqr = q3 - q1
        if iqr < 0.001:
            return points
        mask = (z >= q1 - 1.5 * iqr) & (z <= q3 + 1.5 * iqr)
        return points[mask]


def parse_args():
    """ROS2 인자를 제거한 뒤 argparse로 파싱"""
    filtered_args = remove_ros_args(args=sys.argv)
    parser = argparse.ArgumentParser(description="Pose Estimate Node")
    parser.add_argument(
        "--target_cls",
        type=str,
        default="Can_1",
        help="Target class name for detection (default: Can_1)",
    )
    return parser.parse_args()  # [1:] to skip script name


def main(args=None):
    args = parse_args()
    rclpy.init(args=sys.argv)
    node = PoseEstimateNode(target_cls=args.target_cls)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
