import cv2
import numpy as np
import matplotlib.pyplot as plt


def calibrate_camera(image_points, world_points, image_shape):
    # Convert points to the required shape
    image_points = np.expand_dims(image_points, axis=0)
    world_points = np.expand_dims(world_points, axis=0)

    # Initialize camera matrix with reasonable guesses
    camera_matrix = np.array(
        [[1000, 0, image_shape[0] / 2], [0, 1000, image_shape[1] / 2], [0, 0, 1]],
        dtype=np.float32,
    )
    dist_coeffs = np.zeros((5, 1))

    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        [world_points],
        [image_points],
        image_shape,
        camera_matrix,
        None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
        | cv2.CALIB_FIX_K4
        | cv2.CALIB_FIX_K5
        | cv2.CALIB_ZERO_TANGENT_DIST,
    )

    return camera_matrix, dist_coeffs, rvecs, tvecs


# Perform calibration
image_shape = (image.shape[1], image.shape[0])  # (width, height)
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
    image_points, world_points, image_shape
)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)
print("Rotation Vectors:\n", rvecs)
print("Translation Vectors:\n", tvecs)


def draw_world_points(world_coords, color="r"):
    # Project the world coordinates to image coordinates
    image_coords, _ = cv2.projectPoints(
        world_coords, rvecs[0], tvecs[0], camera_matrix, dist_coeffs
    )
    # Convert from (n, 1, 2) to (n, 2)
    image_coords = image_coords.squeeze()
    plt.scatter(
        image_coords[:, 0], image_coords[:, 1], c=color, marker="o"
    )  # Plot points


def draw_calibrated_points():
    # Plot the image
    # Convert BGR to RGB for plotting with matplotlib
    plt.figure()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    draw_world_points(world_points)
    left_boundary_points = np.array([[0, x, 0] for x in np.linspace(0, length, 10)])
    right_boundary_points = np.array(
        [[width, x, 0] for x in np.linspace(0, length, 10)]
    )
    top_boundary_points = np.array([[x, length, 0] for x in np.linspace(0, width, 10)])
    mid_boundary_points = np.array(
        [[x, length / 2, 0] for x in np.linspace(0, width, 10)]
    )
    bottum_boundary_points = np.array([[x, 0, 0] for x in np.linspace(0, width, 10)])
    draw_world_points(left_boundary_points)
    draw_world_points(right_boundary_points)
    draw_world_points(top_boundary_points)
    draw_world_points(mid_boundary_points)
    draw_world_points(bottum_boundary_points)

    net_points = np.array([[x, length / 2, height] for x in np.linspace(0, width, 10)])
    left_net_points = np.array(
        [[0, length / 2, x] for x in np.linspace(0, 5 * height, 10)]
    )
    right_net_points = np.array(
        [[width, length / 2, x] for x in np.linspace(0, 5 * height, 10)]
    )
    draw_world_points(net_points, "b")
    draw_world_points(left_net_points, "b")
    draw_world_points(right_net_points, "b")
    plt.title("World Coordinates Projected to Image")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")


def draw_labeling_points():
    plt.figure()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb, alpha=0.6)
    x_offset = 10
    y_offset = 10
    plt.scatter(image_points[:, 0], image_points[:, 1], c="b", s=5)
    for image_point, world_point in zip(image_points, world_points):
        plt.text(
            image_point[0] + x_offset,
            image_point[1] + y_offset,
            f"({image_point[0]:.0f}, {image_point[1]:.0f})\n[{world_point[0]:.1f}, {world_point[1]:.1f}, {world_point[2]:.1f}]",
        )
