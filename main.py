import cv2
import numpy as np

from camera import RealSenseCamera
from detector import ObjectDetector
from coordinate import DistanceCalculator


def main():
    camera = RealSenseCamera()
    detector = ObjectDetector()
    distance_calculator = DistanceCalculator()

    camera_started = False

    try:
        camera.start()
        camera_started = True

        print("RealSense D435 started")
        print("Coordinates are relative to the camera optical frame")
        print("Press ESC to quit")

        frame_count = 0

        while True:
            color_frame, depth_frame = camera.get_frames()

            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Assume detector coordinates correspond to the color image
            objects = detector.detect(color_image)

            for obj in objects:
                name = obj["name"]
                confidence = float(obj["confidence"])

                x1, y1, x2, y2 = map(int, obj["bbox"])
                center_x, center_y = map(int, obj["center"])

                position = distance_calculator.get_object_position(
                    depth_frame, center_x, center_y
                )

                cv2.rectangle(
                    color_image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    color_image,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1,
                )

                if position is not None:
                    x = position["x"]
                    y = position["y"]
                    z = position["z"]

                    label = (
                        f"{name} {confidence:.2f} "
                        f"X:{x:.3f} Y:{y:.3f} Z:{z:.3f}m"
                    )

                    # Print only every 30 frames
                    if frame_count % 30 == 0:
                        print(
                            f"{name} ({confidence:.2f}): "
                            f"XYZ= "
                            f"({x:.3f}, {y:.3f}, {z:.3f}) m"
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
                    2,
                )

            cv2.imshow(
                "Object 3D Detection - ESC to exit",
                color_image,
            )

            frame_count += 1

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    finally:
        if camera_started:
            camera.stop()

        cv2.destroyAllWindows()
        print("Camera stopped")


if __name__ == "__main__":
    main()
