import cv2
import numpy as np

from camera import RealSenseCamera
from detector import ObjectDetector
from distance import DistanceCalculator


def main():
    camera = RealSenseCamera()
    detector = ObjectDetector()
    distance_calculator = DistanceCalculator()

    camera.start()

    print("RealSense camera connected")
    print("Press ESC to quit")

    try:
        while True:
            color_frame, depth_frame = camera.get_frames()

            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Detect objects using YOLO
            objects = detector.detect(color_image)

            for obj in objects:
                name = obj["name"]
                confidence = obj["confidence"]

                x1, y1, x2, y2 = obj["bbox"]
                center_x, center_y = obj["center"]

                # Calculate 3D camera-relative position
                position = distance_calculator.get_object_position(
                    depth_frame,
                    center_x,
                    center_y
                )

                cv2.rectangle(
                    color_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    color_image,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

                if position is not None:
                    x = position["x"]
                    y = position["y"]
                    z = position["z"]

                    label = (
                        f"{name} {confidence:.2f} "
                        f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}m"
                    )

                    print(
                        f"{name}: "
                        f"X={x:.3f}m, "
                        f"Y={y:.3f}m, "
                        f"Z={z:.3f}m"
                    )
                else:
                    label = f"{name} {confidence:.2f} XYZ: N/A"

                cv2.putText(
                    color_image,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

            cv2.imshow("Object 3D Position - Press ESC to exit", color_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        camera.stop()
        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()