# -*- coding:utf-8 -*-
"""
@Time:2024-06-03  14:35
@Auth:yuanjinhao
@File:calibration_intrinsic_parameter.py
"""

"""
依据内参矩阵，标定外参矩阵
"""
from calibration import *


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
		calibration.calibration(isshow=True, isprint=True)

	# 计算重投影误差
	calibration.calculate_projection_error()

	# 显示畸变矫正后的图像
	calibration.distortion_correction(isshow=True)

