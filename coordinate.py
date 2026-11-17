import numpy as np
import pyrealsense2 as rs


class DistanceCalculator:

    def __init__(self, sample_radius=5):
        self.sample_radius = sample_radius

    def get_object_position(
            self,
            depth_frame,
            center_x,
            center_y,
            intrinsics=None
    ):

        if depth_frame is None:
            return None

        width = depth_frame.get_width()
        height = depth_frame.get_height()

        center_x = int(center_x)
        center_y = int(center_y)

        if not (0 <= center_x < width and
                0 <= center_y < height):
            return None

        x_min = max(0, center_x - self.sample_radius)
        x_max = min(width, center_x + self.sample_radius + 1)

        y_min = max(0, center_y - self.sample_radius)
        y_max = min(height, center_y + self.sample_radius + 1)

        distances = []

        for y in range(y_min, y_max):
            for x in range(x_min, x_max):

                depth = depth_frame.get_distance(x, y)

                if depth > 0 and np.isfinite(depth):
                    distances.append(depth)

        if not distances:
            return None

        depth = float(np.median(distances))

        if intrinsics is None:
            intrinsics = (
                depth_frame.profile
                .as_video_stream_profile()
                .intrinsics
            )

        point = rs.rs2_deproject_pixel_to_point(
            intrinsics,
            [float(center_x), float(center_y)],
            depth
        )

        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "depth": depth
        }

    def get_object_distance(
            self,
            depth_frame,
            center_x,
            center_y,
            intrinsics=None
    ):

        position = self.get_object_position(
            depth_frame,
            center_x,
            center_y,
            intrinsics
        )

        return None if position is None else position["z"]