import cv2
import numpy as np
from lib.parameters import *


def get_world_coordinates_to_plot():
    court_param = CourtParam()
    points = court_param.court_points
    points = np.append(points, court_param.net_points, axis=0)

    # Define other points
    other_points = np.array([])

    if (len(other_points)) > 0:
        points = np.append(points, other_points, axis=0)
    return points


def get_calibration_matrix(pixel_points, image_shape):
    court_param = CourtParam()

    mask = np.array(
        [False if pixel_point[0] is None else True for pixel_point in pixel_points]
    )
    pixel_points = np.array(pixel_points)[mask]
    world_points = court_param.court_points[mask]

    # Convert points to the required shape
    pixel_points = np.expand_dims(pixel_points, axis=0).astype(np.float32)
    world_points = np.expand_dims(world_points, axis=0).astype(np.float32)

    # Initialize camera matrix with reasonable guesses
    camera_matrix = np.array(
        [
            [CALIBRATION_FOCUS_CONST, 0, image_shape[0] / 2],
            [0, CALIBRATION_FOCUS_CONST, image_shape[1] / 2],
            [0, 0, 1],
        ],
        dtype=np.float32,
    )

    # Calibrate the camera
    dist_coeffs = np.zeros((5, 1))
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [world_points],
        [pixel_points],
        (image_shape[1], image_shape[0]),
        camera_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_ZERO_TANGENT_DIST,
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs


def world_to_pixel(world_points, camera_matrix, dist_coeffs, rvecs, tvecs):
    # Note: due to all points have z=0, OpenCV doesn't know the direction of z axis.
    # It's observed that OpenCV always has opposite z due to the selection of tennis
    # court coordination. So here we mannually flip the z direction of world points.
    world_points[:, 2] *= -1

    image_coords, _ = cv2.projectPoints(
        world_points, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
    )
    # Convert from (n, 1, 2) to (n, 2)
    image_coords = image_coords.squeeze()
    return image_coords
