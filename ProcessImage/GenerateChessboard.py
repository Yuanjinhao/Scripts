# coding:utf-8
"""
Name  : GenerateChessboard.py
Author: yuan.jinhao
Time  : 2024/10/28 下午3:56
Desc  : 生成棋盘格，添加畸变
"""

import cv2
import numpy as np


def generate_image(size, output):
    # 棋盘格尺寸
    rows, cols = size[0] + 1, size[1] + 1
    square_size = 100  # 每个方格的像素大小

    # 创建空白图像
    image = np.ones((3648, 5472), dtype=np.uint8) * 255

    x_offset = int(2736 - square_size * cols / 2)
    y_offset = int(1824 - square_size * rows / 2)

    # 填充棋盘格,
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 1:
                cv2.rectangle(image, (col * square_size + x_offset, row * square_size + y_offset),
                              ((col + 1) * square_size + x_offset, (row + 1) * square_size + y_offset), 0, -1)
    # # 保存和显示图像
    cv2.imwrite(output, image)
    # cv2.imshow('Chessboard', image)
    # cv2.waitKey(0)
    return image


def find_coordinates(image, chessboard_size, show=False):
    # 检测角点
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret:
        # 精细化角点位置
        corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        # 在原图上绘制角点
        if show:
            cv2.drawChessboardCorners(image, chessboard_size, corners2, ret)

            # 显示结果
            cv2.namedWindow('Chessboard', cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
            cv2.resizeWindow('Chessboard', 800, 600)  # 设置窗口大小为 800x600
            cv2.imshow('Chessboard', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        # 角点坐标
        coordinates = corners2.reshape([chessboard_size[1], chessboard_size[0], -1])
        return coordinates
    else:
        print("未找到角点。")


# 生成畸变图像
def apply_radial_distortion(image, k):
    h, w = image.shape[:2]

    # 相机内参
    camera_matrix = np.array([[w, 0, w // 2],
                              [0, h, h // 2],
                              [0, 0, 1]], dtype=np.float32)

    # 畸变系数
    dist_coeffs = np.array([k, 0, 0, 0, 0], dtype=np.float32)

    # 计算新相机矩阵
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    # 应用畸变
    distorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    return distorted_image


if __name__ == "__main__":
    chessboard_size = (16, 22)
    output = './Files/chessboard.png'
    image = generate_image(chessboard_size, output)
    cv2.imshow('Original Image', image)
    coordinates = find_coordinates(image, chessboard_size, False)
    print(f"coordinates:\n {coordinates}")

    # # 设置畸变系数
    k1 = 0.5
    # 应用径向畸变
    distorted_image = apply_radial_distortion(image, k1)
    # distorted_image = barrel_distortion(image, k1)

    cv2.imwrite('./Files/distored.png', distorted_image)
    # 显示原始图像和畸变图像
    cv2.imshow('Distorted Image', distorted_image)

    distorted_coordinates = find_coordinates(distorted_image, chessboard_size, False)
    print(f"distorted_coordinates:\n {distorted_coordinates}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
