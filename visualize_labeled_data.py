import json
import cv2
import argparse
from lib.utils import wait_for_image_visualization_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show the quality of labeled data. Use '<', '>' to switch pictures. Use 'q' to quit."
    )
    parser.add_argument("--start_idx", type=int, default=0, help="start index to plot")
    parser.add_argument(
        "--plot_val",
        action="store_true",
        help="Flag to plot val. Default to plot train",
    )
    args = parser.parse_args()

    label_path = "data/data_val.json" if args.plot_val else "data/data_train.json"

    with open(label_path, "r") as f:
        data = json.load(f)
        i = args.start_idx

        while True:
            # Plot image
            image_path = "data/images/" + data[i]["id"] + ".png"
            image = cv2.imread(image_path)
            image = cv2.putText(
                image,
                image_path,
                (to_int(0.7 * image.shape[1]), to_int(0.95 * image.shape[0])),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(255, 255, 255),
                thickness=1,
            )

            # Plot label
            labeled_points = data[i]["kps"]
            for j in range(len(labeled_points)):
                image = cv2.circle(
                    image,
                    (to_int(labeled_points[j][0]), to_int(labeled_points[j][1])),
                    radius=0,
                    color=(255, 0, 0),
                    thickness=10,
                )
                image = cv2.putText(
                    image,
                    str(j),
                    (to_int(labeled_points[j][0]), to_int(labeled_points[j][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 0),
                    thickness=2,
                )

            cv2.imshow("image", image)

            i = wait_for_image_visualization_key(i, len(data))
