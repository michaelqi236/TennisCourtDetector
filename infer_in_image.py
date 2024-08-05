import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse


from lib.tracknet import BallTrackerNet
from lib.postprocess import *
from lib.parameters import *
from lib.calibration import *
from lib.utils import *


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
        "--plot_3d_to_2d",
        action="store_true",
        help="whether to apply calibration conversion and plot 3d coord in 2d picture",
    )
    parser.add_argument(
        "--plot_2d_to_3d",
        action="store_true",
        help="whether to apply calibration conversion and plot pixel position in 3d. For now we only support to print.",
    )
    parser.add_argument(
        "--debug_likelihood",
        action="store_true",
        help="Debug likelihood distribution for each of the inferred points",
    )
    parser.add_argument(
        "--image_idx", type=int, default=0, help="image idx to start loading"
    )
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("loading model with device", device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    image_idx = args.image_idx
    file_num = len(os.listdir(args.input_path)) if os.path.isdir(args.input_path) else 1
    while image_idx < file_num:
        print("loading image, idx", image_idx)
        image = load_images(args.input_path, image_idx)
        if image.shape[0] / OUTPUT_HEIGHT != image.shape[1] / OUTPUT_WIDTH:
            print(
                "Image size must be proportional to", OUTPUT_HEIGHT, "x", OUTPUT_WIDTH
            )
            exit()

        scale = image.shape[0] / OUTPUT_HEIGHT
        img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
        inp = img.astype(np.float32) / 255.0
        inp = torch.tensor(np.rollaxis(inp, 2, 0))
        inp = inp.unsqueeze(0)

        print("infering image")
        out = model(inp.float().to(device))[0]
        heatmap = F.sigmoid(out).detach().cpu().numpy()

        print("Infer done. Post processing")

        # plot debug likelihood
        if args.debug_likelihood:
            point_idx = 0
            while point_idx < OUTPUT_POINT_NUM:
                point_idx = debug_likelihood_distribution(
                    heatmap,
                    image,
                    scale,
                    thresh=MODEL_OUTPUT_BIN_THRESHOLD,
                    min_radius=CIRCLE_MIN_RADIUS,
                    max_radius=CIRCLE_MAX_RADIUS,
                    point_idx=point_idx,
                )

        # Get inferred points
        inferred_points = []
        point_likelihoods = []
        for i in range(OUTPUT_POINT_NUM):
            x, y, likelihood = postprocess(
                heatmap[i],
                scale,
                thresh=MODEL_OUTPUT_BIN_THRESHOLD,
                min_radius=CIRCLE_MIN_RADIUS,
                max_radius=CIRCLE_MAX_RADIUS,
            )
            if args.use_refine_kps and i not in [8, 12, 9] and x and y:
                x, y = refine_kps(image, to_int(x), to_int(y), scale, REFINE_CROP_SIZE)
            inferred_points.append((x, y))
            point_likelihoods.append(likelihood)

        # Plot labeled points
        if args.plot_label:
            labeled_points = get_labeled_points(args.input_path)
            for i in range(len(labeled_points)):
                image = cv2.circle(
                    image,
                    (to_int(labeled_points[i][0]), to_int(labeled_points[i][1])),
                    radius=0,
                    color=(255, 0, 0),
                    thickness=10,
                )

        # Plot inferred points
        for i in range(len(inferred_points)):
            if inferred_points[i][0] is not None:
                cv2.circle(
                    image,
                    (to_int(inferred_points[i][1]), to_int(inferred_points[i][0])),
                    radius=0,
                    color=(0, 0, 255),
                    thickness=10,
                )
                draw_text_with_background(
                    image,
                    str(f"({point_likelihoods[i]:.2f})"),
                    font=cv2.FONT_HERSHEY_PLAIN,
                    pos=(
                        to_int(inferred_points[i][1]),
                        to_int(inferred_points[i][0] + 15),
                    ),
                    font_scale=1,
                    font_thickness=1,
                    text_color=(255, 255, 255),
                    text_color_bg=(30, 30, 30),
                )

        # Execute conversion
        camera_matrix, dist_coeffs, rvecs, tvecs = get_calibration_matrix(
            inferred_points, image.shape
        )
        # test_conversion(camera_matrix, dist_coeffs, rvecs, tvecs)

        # Calibration camera to get conversion matrix
        if args.plot_3d_to_2d:
            world_points = get_world_coordinates_to_plot()
            pixel_points = world_to_pixel(
                world_points, camera_matrix, dist_coeffs, rvecs, tvecs
            )
            for i in range(len(pixel_points)):
                image = cv2.circle(
                    image,
                    (to_int(pixel_points[i][1]), to_int(pixel_points[i][0])),
                    radius=0,
                    color=(0, 255, 0),
                    thickness=10,
                )

        # Plot 2d to 3d conversion
        if args.plot_2d_to_3d:
            camera_position = get_camera_position(rvecs, tvecs)
            z_candidates = np.linspace(0, camera_position[2], 2)

            pixel_points = np.array(
                [
                    [OUTPUT_HEIGHT * scale / 2, OUTPUT_WIDTH * scale / 2],
                    [OUTPUT_HEIGHT * scale / 2 + 200, OUTPUT_WIDTH * scale / 2 + 200],
                    [OUTPUT_HEIGHT * scale / 2 - 200, OUTPUT_WIDTH * scale / 2 + 200],
                    [OUTPUT_HEIGHT * scale / 2 + 200, OUTPUT_WIDTH * scale / 2 - 200],
                    [OUTPUT_HEIGHT * scale / 2 - 200, OUTPUT_WIDTH * scale / 2 - 200],
                ]
            )

            all_world_points = []
            for pixel_point in pixel_points:
                world_points = pixel_to_world(
                    pixel_point, camera_matrix, dist_coeffs, rvecs, tvecs, z_candidates
                )
                all_world_points.append(world_points)
            all_world_points = np.array(all_world_points)

            # Plot 3d points
            plot_world_points(all_world_points, camera_position)

        else:
            # OpenCV visualization
            if args.output_path:
                cv2.imwrite(args.output_path, image)
            else:
                cv2.imshow("image", image)
                image_idx = wait_for_image_visualization_key(image_idx, file_num)
