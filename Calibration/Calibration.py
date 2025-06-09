# -*- coding:utf-8 -*-
"""
@Time:2024-05-30  16:02
@Auth:yuanjinhao
@File:calibration.py
"""
import time

"""
1.标定内参矩阵与畸变向量，也可以初步输出各标定板的旋转向量和平移向量（不准确）
2.通过标定的内参矩阵和畸变向量，为某一张图像单独标定外参矩阵（准确）
"""

import os
import cv2
import numpy as np
from tqdm import tqdm


class Calibration:
    def __init__(self, chessboard_size, square_size, image_size, images_path, ):
        """
        :param chessboard_size: 标定板的有效标定角点
        :param square_size: 标定板的方格尺寸
        :param images_path: 标定图像路径列表
        """
        self.images_path = images_path
        self.chessboard_size = chessboard_size
        self.image_size = image_size

        # 初始化用于存储角点位置的数据结构
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

        # 准备存储角点在图像中的位置的数据结构
        self.imgpoints = []  # 存储找到的角点在当前图像中的坐标
        self.objpoints = []  # 存储标定棋盘格在世界空间中的坐标

    # 相机标定
    def calibration(self, isshow=False, isprint=False):
        """
        :param isshow: 是否显示标定图像
        :param isprint: 是否打印
        """
        for image_path in tqdm(self.images_path, desc="Calibration:  "):
            src_img = cv2.imread(image_path)

            # 转换为灰度图
            gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)

            # 寻找棋盘格角点
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                # 优化角点位置到亚像素级别
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

                # 将找到的角点添加到全局列表中
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners2)

            # 是否可视化检测到的棋盘格
            if isshow:
                cv2.drawChessboardCorners(src_img, self.chessboard_size, corners2, ret)
                show('Detected Corners', src_img, (1268, 952))
                cv2.waitKey(500)  # 等待500毫秒，然后继续处理下一个图像

        # 使用找到的棋盘格角点和世界坐标计算相机参数
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
            cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_size, None, None, criteria=criteria)

        # 根据旋转向量转换为旋转矩阵
        self.rtxs = []
        for rvec in self.rvecs:
            rvec = np.array(rvec.reshape(-1))
            rtx, _ = cv2.Rodrigues(rvec)
            self.rtxs.append(rtx)
        if isprint:
            print("Calibration Result:")
            print("Camera Matrix:\n", self.mtx)
            print("Distortion Coefficients:\n", self.dist)
            print("Rotation Vectors:\n", self.rvecs)
            print("Translation Vectors:\n", self.tvecs)
            print("Rotation Matrixs:\n", self.rtxs)

        return self.mtx, self.dist, self.rvecs, self.tvecs, self.rtxs

    # 显示畸变矫正后的标定图像
    def distortion_correction(self, isshow=False):
        mapx, mapy = cv2.initUndistortRectifyMap(self.mtx, self.dist, None, None, self.image_size, cv2.CV_16SC2)

        for image_path in tqdm(self.images_path, desc="Distortion Correction:  "):
            src_img = cv2.imread(image_path)

            dst_img = cv2.remap(src_img, mapx, mapy, cv2.INTER_LINEAR)

            # 显示矫正后的标定板图像
            if isshow:
                difference = src_img - dst_img
                show('Src Image', src_img)
                show('Calibrated Chessboard Image', dst_img)
                show('Difference Image', difference)

                # 等待用户关闭窗口
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    # 计算重投影误差
    def calculate_projection_error(self):
        Errors = []
        for i in range(len(self.objpoints)):
            imgpoints2, _ = cv2.projectPoints(self.objpoints[i], self.rvecs[i], self.tvecs[i], self.mtx, self.dist)
            error = cv2.norm(self.imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)

            Errors.append(error)

        print(" Every Image's Reprojection Error:\n ", Errors)


def show(window_name, img, window_size=(634, 476)):
    cv2.namedWindow(window_name, 0)
    cv2.resizeWindow(window_name, window_size)
    cv2.imshow(window_name, img)


def point2world(point2D, rotation_matrix, transfor_matrix, camera_matrix, height):
    point3D = []
    point2D = (np.array(point2D, dtype='float32')).reshape(-1, 2)
    numPts = point2D.shape[0]
    point2D_op = np.hstack((point2D, np.ones((numPts, 1))))
    rMat_inv = np.linalg.inv(rotation_matrix)
    kMat_inv = np.linalg.inv(camera_matrix)
    for point in range(numPts):
        uvPoint = point2D_op[point, :].reshape(3, 1)
        tempMat = np.matmul(rMat_inv, kMat_inv)
        tempMat1 = np.matmul(tempMat, uvPoint).reshape(-1)
        tempMat2 = np.matmul(rMat_inv, transfor_matrix)
        Zc = (height + tempMat2[2]) / tempMat1[2]
        p = tempMat1 * Zc - tempMat2
        point3D.append(p)

    return point3D


def point_distance(points):
    point_0 = points[0]
    point_1 = points[1]

    return ((point_0[0] - point_1[0]) ** 2 + (point_0[1] - point_1[1]) ** 2) ** (1 / 2)


# 单独标定外参矩阵
def calibration_external_parameter(gray, mtx, dist, chessboard_size, square_size):
    # 初始化用于存储角点位置的数据结构
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 优化角点位置到亚像素级别
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    _, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

    rvec = np.array(rvec.reshape(-1))
    rtx, _ = cv2.Rodrigues(rvec)

    return rvec, tvec, rtx


# 去除畸变
def rectify_image(src_img, camera_matrix, dist_coeffs, img_shape):
    # 计算无畸变和修正转换关系
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, None, img_shape[::-1], cv2.CV_16SC2)

    dst_img = cv2.remap(src_img, mapx, mapy, cv2.INTER_LINEAR)

    return dst_img


# 引入畸变,在原图上计算测量点坐标
def distor_points1(points, camera_matrix, distortion_coeffs, image_shape):
    # image_shape = [3648, 5472]
    fx = camera_matrix[0][0]
    fy = camera_matrix[1][1]
    ux = camera_matrix[0][2]
    uy = camera_matrix[1][2]

    k1 = distortion_coeffs[0]
    k2 = distortion_coeffs[1]
    k3 = distortion_coeffs[4]
    p1 = distortion_coeffs[2]
    p2 = distortion_coeffs[3]
    new_points = []
    for point in points:
        x_corrected = (point[0] - ux) / fx
        y_corrected = (point[1] - uy) / fy

        r_2 = x_corrected ** 2 + y_corrected ** 2

        delta_Ra = 1 + k1 * r_2 + k2 * r_2 * r_2 + k3 * r_2 * r_2 * r_2
        delta_Rb = 1

        delta_Tx = 2 * p1 * x_corrected * y_corrected + p2 * (r_2 + 2 * x_corrected * x_corrected)
        delta_Ty = p1 * (r_2 + 2 * y_corrected * y_corrected) + 2 * p2 * x_corrected * y_corrected

        x_distortion = x_corrected * delta_Ra * delta_Rb + delta_Tx
        y_distortion = y_corrected * delta_Ra * delta_Rb + delta_Ty

        x_distortion = x_distortion * fx + ux + (ux - image_shape[1] / 2)
        y_distortion = y_distortion * fy + uy + (uy - image_shape[0] / 2)

        # x_distortion = point[0] + (ux - image_shape[1] / 2)
        # y_distortion = point[1] + (uy - image_shape[0] / 2)

        new_points.append([x_distortion, y_distortion])

    return new_points


def get_corners(gray, chessboard_size):
    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        # 优化角点位置到亚像素级别
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                   criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

    return corners


if __name__ == "__main__":
    chessboard_size = (11, 8)
    square_size = 10
    image_size = (5472, 3648)
    dir_path = './images/testicalibrationplate0603'
    images_name = sorted(os.listdir(dir_path))
    images_path = [os.path.join(dir_path, image_name) for image_name in images_name]
    # 相机标定
    calibration = Calibration(chessboard_size, square_size, image_size, images_path)
    camera_matrix, distortion_coefficient, rotation_vectors, translation_vectors, rotation_matrixs = \
        calibration.calibration(isshow=False, isprint=True)

    # 计算重投影误差
    calibration.calculate_projection_error()

    # 显示畸变矫正后的图像
    calibration.distortion_correction(isshow=True)

    # # 重新标定外参
    # image_path = dir_path + '/1.bmp'
    # src_img = cv2.imread(image_path)
    #
    # new_rotation_vector, new_translation_vector, new_rotation_matrix = \
    #     calibration_external_parameter(src_img, camera_matrix, distortion_coefficient, chessboard_size, square_size)
    # print("New Rotation Vectors:\n", new_rotation_vector)
    # print("New Translation Vectors:\n", new_translation_vector)
    # print("New Rotation Matrixs:\n", new_rotation_matrix)
    #
    # src_img = rectify_image(src_img, camera_matrix, distortion_coefficient)
    # corners = get_corners(src_img, chessboard_size).reshape(-1, 2, 2)
    # for point2ds in corners:
    #     points3d = point2world(point2ds, new_rotation_matrix, new_translation_vector.reshape(-1), camera_matrix, 0)
    #     # points3d = point2world(point2ds, rotation_matrixs[0], translation_vectors[0].reshape(-1), camera_matrix, 0)
    #     distance = point_distance(points3d)
    #     print(f"true distance is {distance} mm")
