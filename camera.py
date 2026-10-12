import pyrealsense2 as rs


class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(
            rs.stream.color,
            width,
            height,
            rs.format.bgr8,
            fps,
        )

        self.config.enable_stream(
            rs.stream.depth,
            width,
            height,
            rs.format.z16,
            fps,
        )

        # Align depth pixels to the color image geometry
        self.align = rs.align(rs.stream.color)

        self.profile = None
        self.depth_scale = None
        self.started = False

    def start(self):
        if self.started:
            return

        self.profile = self.pipeline.start(self.config)

        depth_sensor = (
            self.profile.get_device().first_depth_sensor()
        )
        self.depth_scale = depth_sensor.get_depth_scale()
        self.started = True

        print(f"Depth scale: {self.depth_scale} meters")

        # Debugging the actual camera configuration
        for stream in self.profile.get_streams():
            video = stream.as_video_stream_profile()
            print(
                f"Stream: {stream.stream_type()}, "
                f"resolution: {video.width()}x{video.height()}"
            )

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

        if not (0 <= x < width and 0 <= y < height):
            return None

        distance = depth_frame.get_distance(int(x), int(y))

        if distance <= 0:
            return None

        return distance

    def stop(self):
        if self.started:
            try:
                self.pipeline.stop()
            finally:
                self.started = False
                self.profile = None
                print("RealSense camera stopped")
