import cv2
import numpy as np
import json
from scipy.spatial import distance
from lib.utils import line_intersection, wait_for_image_visualization_key, to_int
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.parameters import *


def calculate_edge_likelihood_map(heatmap):
    # Generate edge likelihood map
    edge_thickness = 15
    edge_likelihood_map = cv2.Canny(heatmap, 50, 150)
    edge_likelihood_map = cv2.GaussianBlur(
        edge_likelihood_map, (edge_thickness, edge_thickness), 255
    )
    edge_likelihood_map = (
        edge_likelihood_map / max(1e-5, np.max(edge_likelihood_map)) * 255
    ).astype(np.uint8)
    # cv2.imshow("blured edge_likelihood_map", edge_likelihood_map)

    return edge_likelihood_map


def calculate_binary_heatmap(heatmap, thresh):
    blurred_heatmap = cv2.GaussianBlur((heatmap * 255).astype(np.uint8), (15, 15), 5)
    ret, binary_heatmap = cv2.threshold(
        blurred_heatmap, thresh * 255, 255, cv2.THRESH_BINARY
    )
    return binary_heatmap


def calculate_circle_likelihood(edge_likelihood_map, x, y, r):
    # Get weighted likelihood on the circle
    mask = np.zeros_like(edge_likelihood_map)
    cv2.circle(
        mask,
        (to_int(y), to_int(x)),
        radius=to_int(r),
        color=255,
        thickness=1,
    )
    weighted_likelihood = np.mean(edge_likelihood_map[mask == 255]) / 255

    return weighted_likelihood


def refine_kps(img, x_ct, y_ct, scale, crop_size):
    refined_x_ct, refined_y_ct = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    x_min = int(max(x_ct - crop_size * scale, 0))
    x_max = int(min(img_height, x_ct + crop_size * scale))
    y_min = int(max(y_ct - crop_size * scale, 0))
    y_max = int(min(img_width, y_ct + crop_size * scale))

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)

    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = inters[1]
                new_y_ct = inters[0]
                if (
                    new_x_ct >= 0
                    and new_x_ct < img_crop.shape[0]
                    and new_y_ct >= 0
                    and new_y_ct < img_crop.shape[1]
                ):
                    refined_x_ct = x_min + new_x_ct
                    refined_y_ct = y_min + new_y_ct
    return refined_x_ct, refined_y_ct


def detect_lines(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    lines = np.squeeze(lines)
    if len(lines.shape) > 0:
        if len(lines) == 4 and not isinstance(lines[0], np.ndarray):
            lines = [lines]
    else:
        lines = []
    return lines


def merge_lines(lines):
    lines = sorted(lines, key=lambda item: item[0])
    mask = [True] * len(lines)
    new_lines = []

    for i, line in enumerate(lines):
        if mask[i]:
            for j, s_line in enumerate(lines[i + 1 :]):
                if mask[i + j + 1]:
                    x1, y1, x2, y2 = line
                    x3, y3, x4, y4 = s_line
                    dist1 = distance.euclidean((x1, y1), (x3, y3))
                    dist2 = distance.euclidean((x2, y2), (x4, y4))
                    if dist1 < 20 and dist2 < 20:
                        line = np.array(
                            [
                                to_int((x1 + x3) / 2),
                                to_int((y1 + y3) / 2),
                                to_int((x2 + x4) / 2),
                                to_int((y2 + y4) / 2),
                            ],
                            dtype=np.int32,
                        )
                        mask[i + j + 1] = False
            new_lines.append(line)
    return new_lines


def get_labeled_points(input_path):
    image_name = input_path.split(".")[0].split("/")[-1]

    with open("data/data_train.json", "r") as f:
        data = json.load(f)
        for i in range(len(data)):
            if image_name == data[i]["id"]:
                return data[i]["kps"]

    with open("data/data_val.json", "r") as f:
        data = json.load(f)
        for i in range(len(data)):
            if image_name == data[i]["id"]:
                return data[i]["kps"]

    return []


def postprocess(heatmap, scale, thresh, min_radius, max_radius):
    binary_heatmap = calculate_binary_heatmap(heatmap, thresh)
    edge_likelihood_map = calculate_edge_likelihood_map(binary_heatmap)

    # Calculate Hough circles
    circles = cv2.HoughCircles(
        binary_heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,  # resolution
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None, None, None

    # Calculate the likelihood of the first circle
    y_pred = circles[0][0][0]
    x_pred = circles[0][0][1]
    hough_radius = circles[0][0][2]
    likelihood = calculate_circle_likelihood(
        edge_likelihood_map, x_pred, y_pred, hough_radius
    )

    if likelihood < CIRCLE_LIKELIHOOD_THRESHOLD:
        return None, None, None

    return x_pred * scale, y_pred * scale, likelihood


def debug_likelihood_distribution(
    heatmap, image, scale, thresh, min_radius, max_radius, point_idx
):
    binary_heatmap = calculate_binary_heatmap(heatmap[point_idx], thresh)
    edge_likelihood_map = calculate_edge_likelihood_map(binary_heatmap)

    # Draw edge_likelihood_map
    alpha = 0.3  # Transparency
    mask = alpha + (1 - alpha) * edge_likelihood_map / 255
    mask = cv2.resize(
        mask,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = np.stack([mask] * 3, axis=-1)
    image_to_draw = (image * mask).astype(np.uint8)

    # Get Hough circles
    circles = cv2.HoughCircles(
        binary_heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,  # resolution
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        for i in range(circles.shape[1]):
            y = circles[0][i][0]
            x = circles[0][i][1]
            radius = circles[0][i][2]
            likelihood = calculate_circle_likelihood(edge_likelihood_map, x, y, radius)
            x *= scale
            y *= scale
            radius *= scale

            # Draw hough circles
            image_to_draw = cv2.circle(
                image_to_draw,
                (to_int(y), to_int(x)),
                radius=to_int(radius),
                color=(0, 0, 255),
                thickness=1,
            )
            # Draw circle radius
            image_to_draw = cv2.putText(
                image_to_draw,
                str(to_int(radius)),
                (to_int(y), to_int(x)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.4,
                color=(255, 255, 255),
                thickness=1,
            )
            # Draw circle likelihood
            image_to_draw = cv2.putText(
                image_to_draw,
                str(f"({likelihood:.2f})"),
                (to_int(y), to_int(x + 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=1,
            )

    # Draw ruler boxes
    ruler_size = 20
    start_point = (0, 0)
    end_point = (to_int(ruler_size * scale), to_int(ruler_size * scale))
    image_to_draw = cv2.rectangle(
        image_to_draw,
        start_point,
        end_point,
        color=(0, 0, 255),
        thickness=1,
    )
    image_to_draw = cv2.putText(
        image_to_draw,
        str(20),
        end_point,
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.4,
        color=(255, 255, 255),
        thickness=1,
    )

    # Show plot
    cv2.imshow("image", image_to_draw)

    return wait_for_image_visualization_key(point_idx, OUTPUT_POINT_NUM)


def plot_world_points(all_world_points, camera_position):
    """
    @input all_world_points: [M, N, 3]
    @input camera_position: [3]
    """

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(camera_position[0], camera_position[1], camera_position[2], "o")
    for world_points in all_world_points:
        ax.plot(world_points[:, 0], world_points[:, 1], world_points[:, 2], "-")

    court_param = CourtParam()
    court_surface = [
        [
            court_param.court_points[0],
            court_param.court_points[1],
            court_param.court_points[3],
            court_param.court_points[2],
        ]
    ]
    poly3d = Poly3DCollection(
        court_surface,
        facecolors="cyan",
        linewidths=1,
        edgecolors="r",
        alpha=0.25,
    )
    ax.add_collection3d(poly3d)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.axis("equal")
    plt.show()
    plt.close(fig)
    exit()
