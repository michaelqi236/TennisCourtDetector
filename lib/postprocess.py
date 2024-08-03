import cv2
import numpy as np
import json
from scipy.spatial import distance
from lib.utils import line_intersection, wait_for_image_visualization_key


def postprocess(heatmap, scale, low_thresh=0.6, min_radius=10, max_radius=30):
    # x is vertical and y is horizontal
    x_pred, y_pred, hough_radius, likelihood = None, None, None, 0
    ret, binary_heatmap = cv2.threshold(
        (heatmap * 255).astype(np.uint8), low_thresh * 255, 255, cv2.THRESH_BINARY
    )

    circles = cv2.HoughCircles(
        binary_heatmap,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        y_pred = circles[0][0][0]
        x_pred = circles[0][0][1]
        hough_radius = circles[0][0][2]

        if (
            x_pred >= 0
            and x_pred < heatmap.shape[0]
            and y_pred >= 0
            and y_pred < heatmap.shape[1]
        ):
            likelihood = heatmap[int(x_pred + 0.5)][int(y_pred + 0.5)]

    return (
        x_pred * scale[0],
        y_pred * scale[1],
        likelihood,
        (int(hough_radius * scale[0] + 0.5), int(hough_radius * scale[1] + 0.5)),
    )


def refine_kps(img, x_ct, y_ct, crop_size=40):
    refined_x_ct, refined_y_ct = x_ct, y_ct

    img_height, img_width = img.shape[:2]
    x_min = max(x_ct - crop_size, 0)
    x_max = min(img_height, x_ct + crop_size)
    y_min = max(y_ct - crop_size, 0)
    y_max = min(img_width, y_ct + crop_size)

    img_crop = img[x_min:x_max, y_min:y_max]
    lines = detect_lines(img_crop)

    if len(lines) > 1:
        lines = merge_lines(lines)
        if len(lines) == 2:
            inters = line_intersection(lines[0], lines[1])
            if inters:
                new_x_ct = int(inters[1])
                new_y_ct = int(inters[0])
                if (
                    new_x_ct > 0
                    and new_x_ct < img_crop.shape[0]
                    and new_y_ct > 0
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
                                int((x1 + x3) / 2),
                                int((y1 + y3) / 2),
                                int((x2 + x4) / 2),
                                int((y2 + y4) / 2),
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


def plot_likelihood_distribution(heatmap, image, scale):
    i = 0
    while i < 14:
        # Draw mask
        alpha = 0.2
        mask = alpha + (1 - alpha) * heatmap[i].astype(np.float32)
        mask = cv2.resize(
            mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
        )
        mask = np.stack([mask] * 3, axis=-1)
        image_to_draw = (image * mask).astype(np.uint8)

        # Draw hough circles
        x_pred, y_pred, likelihood, hough_radius = postprocess(
            heatmap[i], scale, low_thresh=0.6, max_radius=25
        )

        image_to_draw = cv2.ellipse(
            image_to_draw,
            (int(y_pred + 0.5), int(x_pred + 0.5)),
            hough_radius,
            angle=0,
            startAngle=0,
            endAngle=360,
            color=(0, 0, 255),
            thickness=1,
        )

        # Show plot
        cv2.imshow("image", image_to_draw)
        i = wait_for_image_visualization_key(i, 14)
