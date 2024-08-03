import cv2
import numpy as np
import torch
import torch.nn.functional as F
import argparse

from lib.tracknet import BallTrackerNet
from lib.postprocess import postprocess, refine_kps
from lib.homography import get_trans_matrix, refer_kps


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='path to model')
    parser.add_argument('--input_path', type=str, help='path to input image')
    parser.add_argument('--output_path', type=str, help='path to output image')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--use_homography', action='store_true', help='whether to use homography postprocessing')
    args = parser.parse_args()

    model = BallTrackerNet(out_channels=15)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device', device)
    model = model.to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print('model loaded')
    
    OUTPUT_WIDTH = 640
    OUTPUT_HEIGHT = 360

    image = cv2.imread(args.input_path)
    scale = [image.shape[0]/OUTPUT_HEIGHT, image.shape[1]/OUTPUT_WIDTH]
    img = cv2.resize(image, (OUTPUT_WIDTH, OUTPUT_HEIGHT))
    inp = (img.astype(np.float32) / 255.)
    inp = torch.tensor(np.rollaxis(inp, 2, 0))
    inp = inp.unsqueeze(0)
    print('image loaded')

    out = model(inp.float().to(device))[0]
    pred = F.sigmoid(out).detach().cpu().numpy()
    print('image inferred')

    points = []
    for kps_num in range(14):
        heatmap = (pred[kps_num]*255).astype(np.uint8)
        # cv2.imshow('heatmap', heatmap)
        # cv2.waitKey(0)
        x_pred, y_pred = postprocess(heatmap, scale, low_thresh=170, max_radius=25)
        if args.use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(image, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))

    if args.use_homography:
        print('post processing by homography')
        matrix_trans = get_trans_matrix(points)
        if matrix_trans is not None:
            points = cv2.perspectiveTransform(refer_kps, matrix_trans)
            points = [np.squeeze(x) for x in points]

    for j in range(len(points)):
        if points[j][0] is not None:
            image = cv2.circle(image, (int(points[j][0]), int(points[j][1])),
                               radius=0, color=(0, 0, 255), thickness=10)
            image = cv2.putText(image, str(j), (int(points[j][0]), int(points[j][1])), 
                                cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, color=(0, 0, 0),thickness=2)

    if args.output_path:
        cv2.imwrite(args.output_path, image)
    else :
        cv2.imshow('image', image)
        cv2.waitKey(0)