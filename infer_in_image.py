import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from lib.tracknet import BallTrackerNet
from lib.postprocess import (
    postprocess,
    refine_kps,
    get_labeled_points,
    plot_likelihood_distribution,
)
from lib.parameters import *
from lib.calibration import *


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="model/model_tennis_court_det.pt",
        help="path to model",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="path to input image",
    )
    parser.add_argument("--output_path", type=str, help="path to output image")
    parser.add_argument(
        "--use_refine_kps",
        action="store_true",
        help="whether to use refine kps postprocessing",
    )
    parser.add_argument(
        "--plot_label", action="store_true", help="whether to plot the train/val label"
    )
    parser.add_argument(
        "--plot_world_3d_coord",
        action="store_true",
        help="whether to apply 2d to 3d conversion and plot 3d coordinates",
    )
    parser.add_argument(
        "--debug_likelihood",
        action="store_true",
        help="Debug likelihood distribution for each of the inferred points",
    )
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model with device", device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("loading image")
    image = cv2.imread(args.input_path)
    scale = [image.shape[0] / OUTPUT_HEIGHT, image.shape[1] / OUTPUT_WIDTH]
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = img.astype(np.float32) / 255.0
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)

    print("infering image")
    out = model(inp.float().to(device))[0]
    heatmap = F.sigmoid(out).detach().cpu().numpy()

    print("plotting")
    # plot debug likelihood
    if args.debug_likelihood:
        plot_likelihood_distribution(heatmap, image, scale)

    # Get inferred points
    inferred_points = []
    point_likelihoods = []
    for i in range(14):
        x_pred, y_pred, likelihood, hough_radius = postprocess(
            heatmap[i], scale, low_thresh=0.6, max_radius=25
        )
        if args.use_refine_kps and i not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(x_pred), int(y_pred))
        inferred_points.append((x_pred, y_pred))
        point_likelihoods.append((likelihood, hough_radius))

    if args.plot_label:
        labeled_points = get_labeled_points(args.input_path)
        for i in range(len(labeled_points)):
            image = cv2.circle(
                image,
                (int(labeled_points[i][0]), int(labeled_points[i][1])),
                radius=0,
                color=(255, 0, 0),
                thickness=10,
            )

    # Plot inferred points
    for i in range(len(inferred_points)):
        if inferred_points[i][0] is not None:
            image = cv2.circle(
                image,
                (int(inferred_points[i][1]), int(inferred_points[i][0])),
                radius=0,
                color=(0, 0, 255),
                thickness=10,
            )
            image = cv2.putText(
                image,
                str(i),
                (int(inferred_points[i][1]), int(inferred_points[i][0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(0, 0, 0),
                thickness=2,
            )
            image = cv2.ellipse(
                image,
                (int(inferred_points[i][1]), int(inferred_points[i][0])),
                point_likelihoods[i][1],
                angle=0,
                startAngle=0,
                endAngle=360,
                color=(0, 0, 255),
                thickness=1,
            )

    # Plot world 3d coordinates
    if args.plot_world_3d_coord:
        world_points = get_world_coordinates_to_plot()
        camera_matrix, dist_coeffs, rvecs, tvecs = get_calibration_matrix(
            inferred_points, image.shape, CALIBRATION_OUTLIER_DROP_NUM
        )
        pixel_points = world_to_pixel(
            world_points, camera_matrix, dist_coeffs, rvecs, tvecs
        )
        for i in range(len(pixel_points)):
            image = cv2.circle(
                image,
                (int(pixel_points[i][1]), int(pixel_points[i][0])),
                radius=0,
                color=(0, 255, 0),
                thickness=10,
            )

    # Plot
    if args.output_path:
        cv2.imwrite(args.output_path, image)
    else:
        cv2.imshow("image", image)
        cv2.waitKey(0)
