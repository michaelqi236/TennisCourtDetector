import numpy as np
from sympy import Line
import sympy
import cv2
import os


def to_int(x: float):
    return int(x + 0.5)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = to_int(center[0]), to_int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point


def is_point_in_image(x, y, input_width=1280, input_height=720):
    res = False
    if x and y:
        res = (x >= 0) and (x <= input_width) and (y >= 0) and (y <= input_height)
    return res


def wait_for_image_visualization_key(idx, max_idx):
    while idx < max_idx:
        key = cv2.waitKey(0)
        if key == ord("q"):
            cv2.destroyAllWindows()
            exit()
        if key == ord(","):
            return max(0, idx - 1)
        elif key == ord("."):
            if idx + 1 == max_idx:
                cv2.destroyAllWindows()
            return idx + 1


def draw_text_with_background(
    img,
    text,
    font,
    pos,
    font_scale,
    font_thickness,
    text_color=(0, 255, 0),
    text_color_bg=(0, 0, 0),
):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    buffer = 5
    cv2.rectangle(
        img,
        (x - buffer, y - buffer),
        (x + text_w + buffer, y + text_h + buffer),
        text_color_bg,
        -1,
    )
    cv2.putText(
        img,
        text,
        (x, y + text_h + font_scale - 1),
        font,
        font_scale,
        text_color,
        font_thickness,
    )

    return img


def load_images(input_path, image_idx):
    image = None
    if os.path.isdir(input_path):
        list_dir = os.listdir(input_path)
        file_num = len(list_dir)
        while image_idx < file_num:
            filename = list_dir[image_idx]
            file_path = os.path.join(input_path, filename)
            if not os.path.isfile(file_path) and file_path.lower().endswith(
                (".png", ".jpg", ".jpeg")
            ):
                continue

            image = cv2.imread(file_path)
            if image is not None:
                break

    elif os.path.isfile(input_path):
        if not input_path.lower().endswith((".png", ".jpg", ".jpeg")):
            return None
        image = cv2.imread(input_path)

    return image
