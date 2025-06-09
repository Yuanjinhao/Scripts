# coding:utf-8
"""
Name  : SelfCalibration.py
Author: yuan.jinhao
Time  : 2024/10/29 下午3:11
Desc  :自构标定法，先找出棋盘格的畸变中心，将畸变中心的周围四个角点看作畸变点与理想点重合，可以得到理想点的坐标间距，
       使用该坐标间距即可重构出理想棋盘格，理想棋盘格与提取的像素棋盘格即可获得畸变参数k1；再使用使用畸变模型Xi = Xd * (1 + k1 * r**2),
       式中Xi为理想坐标点，Xd为畸变坐标点，r为畸变坐标点畸变中心的距离；使用畸变模型可以得到新的理想坐标点，可以计算得到标定误差。
"""

import cv2
import numpy as np
import math
from scipy.optimize import least_squares


# 寻找标定板中的角点
def find_coordinates(image, chessboard_size, show=False):
    # 检测角点
    ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)

    if ret:
        # 精细化角点位置
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
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
        coordinates = corners2.reshape([chessboard_size[1], chessboard_size[0], 2])
        return coordinates
    else:
        print("未找到角点。")


# 寻找3个最接近畸变中心的角点
def find_three_points(points, center_point):
    # 计算每个点与中心点的距离
    distances = np.linalg.norm(points - center_point, axis=1)

    # 获取距离最小的点的索引
    closest_indices = np.argsort(distances)[:3]

    # 返回最接近的点
    return points[closest_indices]


# 计算理想棋盘格的像素距离
def calculate_distance(four_points):
    def cal_distance(point_1, point_2):
        return math.sqrt((point_2[0] - point_1[0]) ** 2 + (point_2[1] - point_1[1]) ** 2)

    dis_1 = cal_distance(four_points[0], four_points[1])
    dis_2 = cal_distance(four_points[0], four_points[2])
    print(f"dis_1: {dis_1}")
    print(f"dis_2: {dis_2}")

    return (dis_1 + dis_2) / 2


# 重建棋盘格,使用右下坐标作为构建参考
def reconstruct_chessboard(distorted_coordinates, closest_points, gap):
    # 离畸变中心最近的点的索引
    closest_point = closest_points[0]
    closest_b = closest_points[1]
    closest_c = closest_points[2]

    # 寻找点在目前坐标系下的索引
    index = np.where((distorted_coordinates == closest_point).all(axis=2))
    index_b = np.where((distorted_coordinates == closest_b).all(axis=2))
    index_c = np.where((distorted_coordinates == closest_c).all(axis=2))

    # 将索引大的为b点，
    if index_b[1].item() > index_c[1].item():
        point_b = closest_b
        point_c = closest_c
        new_index_b = index_b
        new_index_c = index_c
    else:
        point_b = closest_c
        point_c = closest_b
        new_index_b = index_c
        new_index_c = index_b

    if new_index_b[0].item() < index[0].item() and new_index_c[1].item() < index[1].item():
        cal_point_a = closest_point
        cal_point_b = point_c

    elif new_index_b[0].item() > index[0].item() and new_index_c[1].item() < index[1].item():
        cal_point_a = closest_point
        cal_point_b = point_c

    elif new_index_b[1].item() > index[1].item() and new_index_c[0].item() > index[0].item():
        cal_point_a = point_b
        cal_point_b = closest_point

    elif new_index_b[1].item() > index[1].item() and new_index_c[0].item() < index[0].item():
        cal_point_a = point_b
        cal_point_b = closest_point

    theta = np.arctan2(cal_point_a[1] - cal_point_b[1], cal_point_a[0] - cal_point_b[0])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]]),

    # 构建理想棋盘格
    ideal_coordinates = []
    for row_points in distorted_coordinates:
        coordinates = []
        for col_point in row_points:
            temp_index = np.where((distorted_coordinates == col_point).all(axis=2))
            difference_y = temp_index[0].item() - index[0].item()
            difference_x = temp_index[1].item() - index[1].item()

            x_offset = difference_x * gap
            y_offset = difference_y * gap

            point_offset = [x_offset, y_offset]
            rotated_point = np.dot(R, point_offset)

            x = rotated_point[0][0] + closest_point[0]
            y = rotated_point[0][1] + closest_point[1]
            coordinates.append([x, y])
        ideal_coordinates.append(coordinates)
    return np.array(ideal_coordinates)

# 目标函数
def residuals(k1, ideal_points, distorted_points, distored_center):
    """计算残差函数"""
    cx, cy = distored_center[0], distored_center[1]
    residual = []
    for (xi, yi), (xd, yd) in zip(ideal_points, distorted_points):
        r2 = (xd - cx) ** 2 + (yd - cy) ** 2
        xd_est = (xd - cx) * (1 + k1 * r2)
        yd_est = (yd - cy) * (1 + k1 * r2)
        residual.extend(xi - cx - xd_est)
        residual.extend(yi - cy - yd_est)
    return np.array(residual)  # 确保返回一维数组

# 校正图像
def barrel_distortion(image, k1, distored_center):
    h, w = image.shape[:2]
    # 畸变中心点
    cx, cy = distored_center[0], distored_center[1]

    # 创建目标图像
    dst = np.zeros_like(image)


    # 遍历每个像素
    for i in range(h):
        for j in range(w):
            r = (j - cx + 1) ** 2 + (i - cy + 1) ** 2
            v = int((i - cy) * (1 + k1 * r)) + cy
            u = int((j - cx) * (1 + k1 * r)) + cx

            # 边界检查
            if 0 <= u < w and 0 <= v < h:
                dst[v, u] = image[i, j]

    # # 创建网格坐标
    # j, i = np.meshgrid(np.arange(w), np.arange(h))
    #
    # # 计算 r
    # r = (j - cx + 1) ** 2 + (i - cy + 1) ** 2
    #
    # # 计算新坐标
    # v = (i - cy) * (1 + k1 * r) + cy
    # u = (j - cx) * (1 + k1 * r) + cx
    #
    # # 边界检查
    # valid_mask = (0 <= u) & (u < w) & (0 <= v) & (v < h)
    #
    # # 将有效的 u 和 v 转换为整数索引
    # u = u.astype(int)
    # v = v.astype(int)
    #
    # # 更新 dst
    # dst[v[valid_mask], u[valid_mask]] = image[i[valid_mask], j[valid_mask]]

    return dst


if __name__ == "__main__":
    # 加载图像
    image_path = "./Files/distored.png"
    image = cv2.imread(image_path)

    # 寻找角点
    chessborad_size = (8, 11)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    distorted_coordinates = find_coordinates(gray, chessborad_size, False)
    # print(f"distorted_coordinates:\n {distorted_coordinates}")

    # 畸变中心
    image_shape = image.shape
    distorded_center = [int(image_shape[1] / 2), int(image_shape[0] / 2)]

    closest_points = find_three_points(distorted_coordinates.reshape(-1, 2), distorded_center)
    assert len(closest_points) == 3, "Distorded Center is False"
    cv2.circle(image, distorded_center, 20, (0, 255, 0), -1)
    for point in closest_points:
        point_int = list(map(int, point))
        cv2.circle(image, point_int, 20, (0, 0, 255), -1)

    cv2.namedWindow('Image', cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.resizeWindow('Image', 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow('Image', image)

    # 计算理想棋盘格像素距离
    ideal_distance = calculate_distance(closest_points)
    # print(f"ideal_distance: {ideal_distance}")

    # 重构棋盘格，获取理想坐标点
    ideal_coordinates = reconstruct_chessboard(distorted_coordinates, closest_points, ideal_distance)
    # print(f"ideal_coordinates:\n {ideal_coordinates}")

    # 初始猜测
    initial_k1 = 1.0
    # 最小化残差
    result = least_squares(residuals, initial_k1, args=(
        ideal_coordinates.reshape(-1, 2), distorted_coordinates.reshape(-1, 2), distorded_center))
    # 输出结果
    k1_estimated = result.x[0]
    print(f"估计的畸变参数 k1: {k1_estimated}")

    # 校正畸变图像
    ideal_image = barrel_distortion(image, k1_estimated, distorded_center)
    cv2.namedWindow('Ideal_image', cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.resizeWindow('Ideal_image', 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow('Ideal_image', ideal_image)
    #
    # 差值
    diff = ideal_image - image
    cv2.namedWindow('Diff', cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.resizeWindow('Diff', 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow('Diff', diff)

    # 绘制结果图像
    mask = np.zeros(shape=image_shape, dtype='uint8')
    # mask = ideal_image
    cv2.circle(mask, distorded_center, 5, (255, 0, 0), -1)
    id = 0
    for distored_point, ideal_point in zip(distorted_coordinates.reshape(-1, 2), ideal_coordinates.reshape(-1, 2)):
        p_1 = list(map(int, distored_point))
        p_2 = list(map(int, ideal_point))

        cv2.putText(mask, f"{id}", p_1, color=(0, 0, 255), fontScale=2, fontFace=4, thickness=4)
        cv2.putText(mask, f"{id}", [p_2[0] + 30, p_2[1] + 30], color=(0, 255, 0), fontScale=2, fontFace=4, thickness=4)
        cv2.circle(mask, p_1, 50, color=(0, 0, 255), thickness=4)
        cv2.circle(mask, p_2, 5, color=(0, 255, 0), thickness=4)
        id += 1
    cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)  # 创建可调整大小的窗口
    cv2.resizeWindow('Mask', 800, 600)  # 设置窗口大小为 800x600
    cv2.imshow('Mask', mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
