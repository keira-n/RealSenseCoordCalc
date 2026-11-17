import argparse
import time

import numpy as np

from camera import RealSenseCamera
from detector import ObjectDetector
from coordinate import DistanceCalculator


def parse_args():
    parser = argparse.ArgumentParser(description="D435 tomato detection and 3D localization")
    parser.add_argument("--model", default="yolo11n.pt",
                        help="YOLO weights path (use tomato-trained weights for tomato detection)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--target-class", default="tomato",
                        help="Exact class name to localize; use 'all' to disable filtering")
    parser.add_argument("--ros", action="store_true",
                        help="Publish targets transformed into the robot base frame using TF2")
    parser.add_argument("--camera-frame", default="camera_color_optical_frame",
                        help="TF frame of D435 color optical coordinates")
    parser.add_argument("--base-frame", default="base_link",
                        help="Robot base TF frame")
    parser.add_argument("--topic", default="/tomato/target_point")
    parser.add_argument("--log-every", type=int, default=30,
                        help="Print detection status every N frames")
    return parser.parse_args()


class RosTargetPublisher:
    # Publishes camera points after a live TF2 transform into base_frame

    def __init__(self, camera_frame, base_frame, topic):
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.duration import Duration
            from geometry_msgs.msg import PointStamped
            from tf2_ros import Buffer, TransformListener, TransformException
            import tf2_geometry_msgs  # Registers geometry message transforms.
        except ImportError as exc:
            raise RuntimeError(
                "ROS mode requires ROS 2 Foxy Python packages, tf2_ros, "
                "tf2_geometry_msgs, and sourced ROS environment"
            ) from exc

        self.rclpy = rclpy
        self.Duration = Duration
        self.PointStamped = PointStamped
        self.TransformException = TransformException

        class PublisherNode(Node):
            def __init__(node_self):
                super().__init__("tomato_realsense_perception")
                node_self.publisher = node_self.create_publisher(PointStamped, topic, 10)

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = PublisherNode()
        self.camera_frame = camera_frame
        self.base_frame = base_frame
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self.node)
        self.last_tf_warning = 0.0
        self.node.get_logger().info(
            "Publishing target points on %s; TF %s <- %s" %
            (topic, base_frame, camera_frame)
        )

    def publish(self, xyz, class_name, confidence):
        camera_point = self.PointStamped()
        camera_point.header.stamp = self.node.get_clock().now().to_msg()
        camera_point.header.frame_id = self.camera_frame
        camera_point.point.x = float(xyz["x"])
        camera_point.point.y = float(xyz["y"])
        camera_point.point.z = float(xyz["z"])

        try:
            base_point = self.buffer.transform(
                camera_point, self.base_frame,
                timeout=self.Duration(seconds=0.15)
            )
        except self.TransformException as exc:
            now = time.monotonic()
            if now - self.last_tf_warning > 5.0:
                self.node.get_logger().warning(
                    "No usable TF transform %s <- %s: %s. "
                    "Check the measured wrist-camera transform and frame names." %
                    (self.base_frame, self.camera_frame, exc)
                )
                self.last_tf_warning = now
            return False

        self.node.publisher.publish(base_point)
        self.node.get_logger().info(
            "%s %.2f -> base XYZ=(%.3f, %.3f, %.3f) m" %
            (class_name, confidence, base_point.point.x,
             base_point.point.y, base_point.point.z)
        )
        return True

    def spin_once(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if self.rclpy.ok():
            self.rclpy.shutdown()


def main():
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between 0 and 1")
    if args.log_every < 1:
        raise SystemExit("--log-every must be >= 1")

    target_classes = None if args.target_class.lower() == "all" else [args.target_class]
    camera = RealSenseCamera()
    detector = ObjectDetector(args.model)
    distance_calculator = DistanceCalculator()
    ros_publisher = None
    camera_started = False
    frame_count = 0

    try:
        if args.ros:
            ros_publisher = RosTargetPublisher(
                args.camera_frame, args.base_frame, args.topic
            )

        camera.start()
        camera_started = True
        print("RealSense D435 started")
        print("Coordinates: camera optical frame (metres)")
        print("YOLO model:", args.model)
        print("Target class:", target_classes if target_classes else "all classes")
        if args.ros:
            print("ROS2 target topic:", args.topic, "(frame:", args.base_frame + ")")
        print("CLI-only mode; press Ctrl+C to stop")
        print("-" * 60)

        while ros_publisher is None or ros_publisher.rclpy.ok():
            color_frame, depth_frame = camera.get_frames()
            if color_frame is None or depth_frame is None:
                if ros_publisher is not None:
                    ros_publisher.spin_once()
                continue

            color_image = np.asanyarray(color_frame.get_data())
            objects = detector.detect(
                color_image, confidence=args.confidence,
                target_classes=target_classes
            )
            frame_count += 1

            if ros_publisher is not None:
                ros_publisher.spin_once()

            # Only publish the highest-confidence target in each frame
            if objects:
                obj = objects[0]
                center_x, center_y = obj["center"]
                position = distance_calculator.get_object_position(
                    depth_frame, center_x, center_y
                )
                if position is not None and ros_publisher is not None:
                    ros_publisher.publish(position, obj["name"], obj["confidence"])

            if frame_count % args.log_every == 0:
                print("\nFrame:", frame_count)
                if not objects:
                    print("No target detected")
                    continue
                for obj in objects:
                    center_x, center_y = obj["center"]
                    position = distance_calculator.get_object_position(
                        depth_frame, center_x, center_y
                    )
                    if position is None:
                        print("%s (%.2f): depth unavailable" %
                              (obj["name"], obj["confidence"]))
                        continue
                    print(
                        "%s (%.2f) | pixel=(%d, %d) | camera XYZ=(%.3f, %.3f, %.3f) m" %
                        (obj["name"], obj["confidence"], center_x, center_y,
                         position["x"], position["y"], position["z"])
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if camera_started:
            camera.stop()
        if ros_publisher is not None:
            ros_publisher.close()
        print("Camera stopped")


if __name__ == "__main__":
    main()
