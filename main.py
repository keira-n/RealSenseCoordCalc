
import numpy as np

from camera import RealSenseCamera
from detector import ObjectDetector
from coordinate import DistanceCalculator


def main():
    camera = RealSenseCamera()
    detector = ObjectDetector()
    distance_calculator = DistanceCalculator()

    camera_started = False
    frame_count = 0

    try:
        camera.start()
        camera_started = True

        print("RealSense D435 started")
        print("Coordinates: camera optical frame")
        print("Press Ctrl+C to stop")
        print("-" * 60)

        while True:
            color_frame, depth_frame = camera.get_frames()

            if color_frame is None or depth_frame is None:
                continue

            color_image = np.asanyarray(
                color_frame.get_data()
            )

            objects = detector.detect(
                color_image,
                confidence=0.5
            )

            frame_count += 1

            # Print results every 30 frames
            if frame_count % 30 == 0:

                print(f"\nFrame: {frame_count}")

                if not objects:
                    print("No objects detected")
                    continue

                for obj in objects:
                    name = obj["name"]
                    confidence = obj["confidence"]

                    center_x, center_y = obj["center"]

                    position = (
                        distance_calculator.get_object_position(
                            depth_frame,
                            center_x,
                            center_y
                        )
                    )

                    if position is None:
                        print(
                            f"{name} ({confidence:.2f}): "
                            "Depth unavailable"
                        )
                        continue

                    x = position["x"]
                    y = position["y"]
                    z = position["z"]

                    print(
                        f"{name} ({confidence:.2f}) | "
                        f"Pixel: ({center_x}, {center_y}) | "
                        f"XYZ: ({x:.3f}, "
                        f"{y:.3f}, {z:.3f}) m"
                    )

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        if camera_started:
            camera.stop()

        print("Camera stopped")


if __name__ == "__main__":
    main()
