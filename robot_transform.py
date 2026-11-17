import numpy as np


def transform_camera_to_robot(camera_point, transformation_matrix):
    """Transform XYZ metres from camera coordinates into robot-base coordinates.

    Args:
        camera_point: Iterable containing (x, y, z) in metres.
        transformation_matrix: 4x4 homogeneous transform T_robot_camera.

    Returns:
        NumPy array (x, y, z) in the robot base frame, in metres.
    """
    point = np.asarray(camera_point, dtype=float)
    transform = np.asarray(transformation_matrix, dtype=float)

    if point.shape != (3,):
        raise ValueError("camera_point must contain exactly three values")
    if transform.shape != (4, 4):
        raise ValueError("transformation_matrix must have shape (4, 4)")
    if not np.all(np.isfinite(point)) or not np.all(np.isfinite(transform)):
        raise ValueError("point and transformation matrix must be finite")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError("invalid homogeneous transformation last row")

    point_camera = np.append(point, 1.0)
    point_robot = transform @ point_camera
    return point_robot[:3]


if __name__ == "__main__":
    # Identity is a software test only, not a camera-to-robot calibration
    T_robot_camera = np.eye(4)
    tomato_camera = [0.10, 0.05, 0.60]
    tomato_robot = transform_camera_to_robot(tomato_camera, T_robot_camera)
    print("Camera coordinates (m):", tomato_camera)
    print("Robot coordinates (m):", tomato_robot)
