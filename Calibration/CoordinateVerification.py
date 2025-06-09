# -*- coding:utf-8 -*-
"""
@Time:2024/6/5  15:23
@Auth:yuanjinhao
@File:coordinate_verification.py
"""
"""
验证反畸变矫正代码
"""

import cv2
import numpy as np

from calibration import *

camera_matrix = np.array([[1.47398380e+04, 0.00000000e+00, 2.69164907e+03],
                          [0.00000000e+00, 1.47392043e+04, 1.87747596e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
distortion_coefficients = np.array([-6.44795502e-02, 1.26520250e+00, 8.86868577e-05, 6.65685463e-04, 7.55022133e+01])

translation_vectors = np.array([38.12476686, -31.7455947, 753.10446671])
rotation_matrixs = np.array([[0.01284695, -0.99927522, 0.03583275],
                             [0.99979552, 0.01339681, 0.01514749],
                             [-0.01561656, 0.03563082, 0.999243]])

if __name__ == "__main__":
    dir_path = './images/testicalibrationplate0603'
    image_path = dir_path + '/2.bmp'
    src_img = cv2.imread(image_path)
    img_shape = src_img.shape[:2]

    chessboard_size = (11, 8)
    corners1 = get_corners(src_img, chessboard_size).reshape(-1, 2)
    for point2d in corners1:
        cv2.circle(src_img, np.int32(point2d), 5, (0, 0, 255), 4)

    rec_img = rectify_image(src_img, camera_matrix, distortion_coefficients, img_shape)

    corners2 = get_corners(rec_img, chessboard_size).reshape(-1, 2)

    for point2d in corners2:
        cv2.circle(src_img, np.int32(point2d), 3, (0, 255, 0), 3)

    corners3 = distor_points1(corners2, camera_matrix, distortion_coefficients, img_shape)

    for point2d in corners3:
        cv2.circle(src_img, np.int32(point2d), 3, (255, 0, 0), 3)

    show("src_img", src_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
