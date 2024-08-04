import cv2
import numpy as np
from lib.parameters import *
import copy


def get_world_coordinates_to_plot():
    """
    @output points: [N, 3]
    """

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
    """
    @input world_points: [N, 3]
    @output image_coords: [N, 2]
    """

    # Note: due to all points have z=0, OpenCV doesn't know the direction of z axis.
    # It's observed that OpenCV always has opposite z due to the selection of tennis
    # court coordination. So here we mannually flip the z direction of world points.
    # reverted_world_points = world_points.deepcopy()
    reverted_world_points = copy.deepcopy(world_points).astype(np.float32)
    reverted_world_points[:, 2] *= -1

    image_coords = None
    manual = False

    if not manual:
        image_coords, _ = cv2.projectPoints(
            reverted_world_points,
            rvecs[0],
            tvecs[0],
            camera_matrix,
            dist_coeffs,
        )
        # Convert from (n, 1, 2) to (n, 2)
        image_coords = np.squeeze(image_coords, axis=1)
    else:
        R, _ = cv2.Rodrigues(rvecs[0])
        image_coords = camera_matrix @ (R @ reverted_world_points.T + tvecs[0])
        image_coords = image_coords.T
        image_coords = image_coords / image_coords[:, 2:]
        image_coords = image_coords[:, :2]

    return image_coords


def pixel_to_world(pixel_point, camera_matrix, dist_coeffs, rvecs, tvecs, z_candidates):
    """
    @input pixel_point: [2]
    @input z_candidates: [N]
    @output image_coords: [N, 3]
    """

    # 3d conversion
    pixel_point = np.append(pixel_point, [1])
    R, _ = cv2.Rodrigues(rvecs[0])
    R_inv = np.linalg.inv(R)

    A = R_inv @ np.linalg.inv(camera_matrix) @ pixel_point
    B = np.squeeze(R_inv @ tvecs[0])
    den = 1e-3 if abs(A[2]) < 1e-3 else A[2]

    # Due to opposite sign of z axis for the calibration matrixm, we need to calculate
    # reverted world point with -z, then flip the reverted world point.
    world_points = []
    for z in z_candidates:
        p = (-z + B[2]) / den
        reverted_world_point = p * A - B
        world_point = [
            reverted_world_point[0],
            reverted_world_point[1],
            -reverted_world_point[2],
        ]
        world_points.append(world_point)
    world_points = np.array(world_points)

    return world_points


def test_conversion(camera_matrix, dist_coeffs, rvecs, tvecs):
    world_points = np.array([[50, 0, 5]])

    pixel_points = world_to_pixel(
        world_points, camera_matrix, dist_coeffs, rvecs, tvecs
    )
    calculated_world_points = pixel_to_world(
        pixel_points[0], camera_matrix, dist_coeffs, rvecs, tvecs, world_points[0, -1:]
    )

    equal = np.allclose(world_points, calculated_world_points, atol=1e-5)
    print("is equal:", equal)
    print("Ground truth world_points", world_points)
    print("calculated_world_points", calculated_world_points)
