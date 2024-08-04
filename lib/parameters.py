# -*- coding: utf-8 -*-

import numpy as np

# Shape of images for model input
OUTPUT_WIDTH = 640
OUTPUT_HEIGHT = 360
OUTPUT_POINT_NUM = 14

# Hough circle detection
MODEL_OUTPUT_BIN_THRESHOLD = 0.6
CIRCLE_MIN_RADIUS = 10
CIRCLE_MAX_RADIUS = 25
CIRCLE_LIKELIHOOD_THRESHOLD = 0.3

# Line intersction refinement
REFINE_CROP_SIZE = 20

# Calibration matrix initial guess
CALIBRATION_FOCUS_CONST = 1000


# Params of tennis court in [m]
class CourtParam:
    def __init__(self):
        # https://www.harrodsport.com/advice-and-guides/tennis-court-dimensions
        YARD_TO_METER = 0.9144

        # Shapes
        self.length = 26 * YARD_TO_METER
        self.width = 12 * YARD_TO_METER
        self.inner_width = 9 * YARD_TO_METER
        self.box_length = 7 * YARD_TO_METER
        self.box_width = self.inner_width / 2
        self.inner_outer_width_gap = (self.width - self.inner_width) / 2
        self.box_border_length_gap = self.length / 2 - self.box_length

        self.net_height_at_post = 7 / 6 * YARD_TO_METER
        self.net_height_at_center = 1 * YARD_TO_METER
        self.net_post_outside_court_distance = 1 * YARD_TO_METER

        """
               ^
             y |
               |
               0---4----------------6---1
               |   |                |   |
               |   |                |   |
               |   |                |   |
               |   8------12--------9   |
               |   |       |        |   |
            ╓--┼---┼-------┼--------┼---┼--╖
            ║  |   |       |        |   |  ║
            ║  |   |       |        |   |  ║
            ╙--┼---┼-------┼--------┼---┼--╜
               |   |       |        |   |
               |   |       |        |   |
               |   |       |        |   |
               |   10-----13-------11   |
               |   |                |   |
               |   |                |   |
               |   |                |   |
               2---5----------------7---3-------> x
        """

        # Coordinates
        self.court_points = np.array(
            [
                [0, self.length, 0],  # 0
                [self.width, self.length, 0],  # 1
                [0, 0, 0],  # 2
                [self.width, 0, 0],  # 3
                [self.inner_outer_width_gap, self.length, 0],  # 4
                [self.inner_outer_width_gap, 0, 0],  # 5
                [self.inner_outer_width_gap + self.inner_width, self.length, 0],  # 6
                [self.inner_outer_width_gap + self.inner_width, 0, 0],  # 7
                [
                    self.inner_outer_width_gap,
                    self.length - self.box_border_length_gap,
                    0,
                ],  # 8
                [
                    self.width - self.inner_outer_width_gap,
                    self.length - self.box_border_length_gap,
                    0,
                ],  # 9
                [self.inner_outer_width_gap, self.box_border_length_gap, 0],  # 10
                [
                    self.width - self.inner_outer_width_gap,
                    self.box_border_length_gap,
                    0,
                ],  # 11
                [self.width / 2, self.length - self.box_border_length_gap, 0],  # 12
                [self.width / 2, self.box_border_length_gap, 0],  # 13
            ]
        )

        """
               ^
             y |
               |
               ├---┬----------------┬---┐
               |   |                |   |
               |   |                |   |
               |   |                |   |
               |   ├-------┬--------┤   |
               |   |       |        |   |
            0--6---┼-------4--------┼---8--2
            ║  |   |       |        |   |  ║
            ║  |   |       |        |   |  ║
            1--7---┼-------5--------┼---9--3
               |   |       |        |   |
               |   |       |        |   |
               |   |       |        |   |
               |   ├-------┴--------┤   |
               |   |                |   |
               |   |                |   |
               |   |                |   |
               └---┴----------------┴---┴-------> x
        """
        # Net Coordinates
        self.net_points = np.array(
            [
                [
                    -self.net_post_outside_court_distance,
                    self.length / 2,
                    self.net_height_at_post,
                ],  # 0
                [
                    -self.net_post_outside_court_distance,
                    self.length / 2,
                    0,
                ],  # 1
                [
                    self.width + self.net_post_outside_court_distance,
                    self.length / 2,
                    self.net_height_at_post,
                ],  # 2
                [
                    self.width + self.net_post_outside_court_distance,
                    self.length / 2,
                    0,
                ],  # 3
                [
                    self.width / 2,
                    self.length / 2,
                    self.net_height_at_center,
                ],  # 4
                [
                    self.width / 2,
                    self.length / 2,
                    0,
                ],  # 5
                [
                    0,
                    self.length / 2,
                    self.net_height_at_post,
                ],  # 6
                [
                    0,
                    self.length / 2,
                    0,
                ],  # 7
                [
                    self.width,
                    self.length / 2,
                    self.net_height_at_post,
                ],  # 8
                [
                    self.width,
                    self.length / 2,
                    0,
                ],  # 9
            ]
        )
