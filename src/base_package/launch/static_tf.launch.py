# static_tf_launch.py
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            # TCP
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="tcp_tf",
                arguments=[
                    "--x",
                    "0.0",
                    "--y",
                    "0.0",
                    "--z",
                    "0.14",
                    "--qx",
                    "0.5",
                    "--qy",
                    "0.5",
                    "--qz",
                    "0.5",
                    "--qw",
                    "-0.5",
                    "--frame-id",
                    "tool0_controller",
                    "--child-frame-id",
                    "tcp",
                ],
            ),
            # Opti-Track
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="opti_track_tf",
                arguments=[
                    "--x",
                    "0.225",
                    "--y",
                    "0.325",
                    "--z",
                    "0.0",
                    "--qx",
                    "0.5",
                    "--qy",
                    "-0.5",
                    "--qz",
                    "-0.5",
                    "--qw",
                    "0.5",
                    "--frame-id",
                    "base_link",
                    "--child-frame-id",
                    "opti_world",
                ],
            ),
            # Helios Camera
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="helios_tf",
                arguments=[
                    "--x",
                    "0.004061",
                    "--y",
                    "0.05116",
                    "--z",
                    "0.002374",
                    "--qx",
                    "0.02098779",
                    "--qy",
                    "0.00122301",
                    "--qz",
                    "-0.00032638",
                    "--qw",
                    "0.99977891",
                    "--frame-id",
                    "triton_camera",
                    "--child-frame-id",
                    "helios_camera",
                ],
            ),
            # Triton Camera
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="triton_tf",
                arguments=[
                    "--x",
                    "0.3480980129049314",
                    "--y",
                    "1.1366173565306177",
                    "--z",
                    "0.3985054952068591",
                    "--qx",
                    "0.9315636603643267",
                    "--qy",
                    "9.153268858934325e-05",
                    "--qz",
                    "-0.0034850110724279895",
                    "--qw",
                    "0.36356153950632525",
                    "--frame-id",
                    "opti_world",
                    "--child-frame-id",
                    "triton_camera",
                ],
            ),
        ]
    )
