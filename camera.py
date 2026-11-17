import pyrealsense2 as rs
import numpy as np


class RealSenseCamera:

    def __init__(self, width=640, height=480, fps=30):

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps
        )

        self.config.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps
        )

        self.align = rs.align(rs.stream.color)

        self.profile = None
        self.depth_scale = None
        self.intrinsics = None
        self.started = False

    def start(self):

        if self.started:
            return

        self.profile = self.pipeline.start(self.config)

        device = self.profile.get_device()

        depth_sensor = device.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Obtain aligned color camera intrinsics
        color_profile = self.profile.get_stream(
            rs.stream.color
        ).as_video_stream_profile()

        self.intrinsics = color_profile.get_intrinsics()

        self.started = True

        print("RealSense started")
        print(f"Depth scale: {self.depth_scale}")
        print(f"fx: {self.intrinsics.fx}")
        print(f"fy: {self.intrinsics.fy}")
        print(f"cx: {self.intrinsics.ppx}")
        print(f"cy: {self.intrinsics.ppy}")

    def get_frames(self):

        if not self.started:
            raise RuntimeError("Camera has not been started")

        try:
            frames = self.pipeline.wait_for_frames(
                timeout_ms=5000
            )
        except RuntimeError as exc:
            print(f"Frame acquisition failed: {exc}")
            return None, None

        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        return color_frame, depth_frame

    def get_distance(self, depth_frame, x, y):

        if depth_frame is None:
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        x, y = int(x), int(y)

        if not (0 <= x < width and 0 <= y < height):
            return None

        distance = depth_frame.get_distance(x, y)

        if distance <= 0:
            return None

        return distance

    def get_median_distance(
            self, depth_frame, x, y, radius=5
    ):

        if depth_frame is None:
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        x, y = int(x), int(y)

        x1 = max(0, x - radius)
        x2 = min(width, x + radius + 1)

        y1 = max(0, y - radius)
        y2 = min(height, y + radius + 1)

        distances = []

        for py in range(y1, y2):
            for px in range(x1, x2):

                distance = depth_frame.get_distance(
                    px, py
                )

                if distance > 0:
                    distances.append(distance)

        if not distances:
            return None

        return float(np.median(distances))

    def get_3d_point(self, depth_frame, x, y):

        if self.intrinsics is None:
            raise RuntimeError("Camera not initialized")

        distance = self.get_median_distance(
            depth_frame, x, y
        )

        if distance is None:
            return None

        point = rs.rs2_deproject_pixel_to_point(
            self.intrinsics,
            [float(x), float(y)],
            distance
        )

        return {
            "x": point[0],
            "y": point[1],
            "z": point[2]
        }

    def stop(self):

        if self.started:

            try:
                self.pipeline.stop()

            finally:
                self.started = False
                self.profile = None
                self.intrinsics = None

                print("RealSense camera stopped")
