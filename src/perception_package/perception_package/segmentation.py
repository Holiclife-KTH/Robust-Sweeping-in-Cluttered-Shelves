import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from rclpy.publisher import Publisher

from sensor_msgs.msg import Image, PointCloud2
from custom_msgs.msg import BBox, BBoxMultiArray

from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from base_package.manager import ImageManager, Manager

# ── Configuration ──────────────────────────────────────────────
MODEL_PATH = "/home/irol/workspace/Robust-Sweeping-in-Cluttered-Shelves/src/perception_package/resource/260301/weights/best.pt"
CAMERA_TOPIC = "/triton/image_raw"
SEGMENTATION_TOPIC = "/segmentation/image"
CONF_THRESHOLD = 0.5
# ───────────────────────────────────────────────────────────────


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("segmentation_node")

        self._model = YOLO(MODEL_PATH, verbose=False)
        self._conf_threshold = CONF_THRESHOLD

        self.__image_manager = ImageManager(
            node=self,
            subscribed_topics=[
                {"topic_name": CAMERA_TOPIC, "callback": self._camera_callback}
            ],
            published_topics=[{"topic_name": SEGMENTATION_TOPIC}],
        )

        self._bbox_pub = self.create_publisher(
            BBoxMultiArray, "/segmentation/bboxes", qos_profile_system_default
        )

        self.get_logger().info(f"Model  : {MODEL_PATH}")
        self.get_logger().info(f"Sub    : {CAMERA_TOPIC}")
        self.get_logger().info(f"Pub img: {SEGMENTATION_TOPIC}")

    # ── Callback ───────────────────────────────────────────────
    def _camera_callback(self, msg: Image):
        image = self.__image_manager.decode_message(msg, desired_encoding="bgr8")
        if image is None:
            return

        results: Results = self._model.predict(image)[0]
        boxes: Boxes = results.boxes
        names: dict = results.names

        np_xyxy = boxes.xyxy.cpu().numpy()
        np_conf = boxes.conf.cpu().numpy()
        np_cls = boxes.cls.cpu().numpy()

        bbox_msg = BBoxMultiArray()

        for i in range(len(boxes)):
            conf = float(np_conf[i])
            if conf < self._conf_threshold:
                continue

            cls_id = int(np_cls[i])
            cls_name = names[cls_id]
            x1, y1, x2, y2 = map(int, np_xyxy[i])

            bbox_msg.data.append(
                BBox(id=cls_id, cls=cls_name, conf=conf, bbox=[x1, y1, x2, y2])
            )

            label = f"{cls_name} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        seg_msg = self.__image_manager.encode_message(image, encoding="bgr8")
        self.__image_manager.publish(SEGMENTATION_TOPIC, seg_msg)
        self._bbox_pub.publish(bbox_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
