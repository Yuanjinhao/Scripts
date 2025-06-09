# -*- coding:utf-8 -*-
"""
@Time:2024-06-03  14:36
@Auth:yuanjinhao
@File:calibration_external_parameter.py
"""
import cv2

"""
标定内参矩阵
"""
from calibration import *

camera_matrix = np.array([[1.47398380e+04, 0.00000000e+00, 2.69164907e+03],
                          [0.00000000e+00, 1.47392043e+04, 1.87747596e+03],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

distortion_coefficients = np.array([-6.44795502e-02, 1.26520250e+00, 8.86868577e-05, 6.65685463e-04, 7.55022133e+01])

rotation_vectors = np.array([-0.20864218, 0.06288339, 1.23252655])

# 首次标定的外参矩阵
# translation_vectors = np.array([7.58226079, -53.94853564, 760.75422247])
# rotation_matrixs = np.array([[0.3328323, -0.94074719, -0.06494136],
#                              [0.92925292, 0.3154959, 0.19222733],
#                              [-0.16034859, -0.12432642, 0.9791993]])

# 重标定的外参矩阵
translation_vectors = np.array([38.12476686, -31.7455947, 753.10446671])
rotation_matrixs = np.array([[0.01284695, -0.99927522, 0.03583275],
                             [0.99979552, 0.01339681, 0.01514749],
                             [-0.01561656, 0.03563082, 0.999243]])

if __name__ == "__main__":
	# 重新标定外参
	chessboard_size = (11, 8)
	square_size = 10
	image_size = (5472, 3648)
	dir_path = './images/testicalibrationplate0603'
	image_path = dir_path + '/2.bmp'
	src_img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
	img_shape = src_img.shape

	new_rotation_vector, new_translation_vector, new_rotation_matrix = \
		calibration_external_parameter(src_img, camera_matrix, distortion_coefficients, chessboard_size, square_size)
	print("New Rotation Vectors:\n", new_rotation_vector)
	print("New Translation Vectors:\n", new_translation_vector)
	print("New Rotation Matrixs:\n", new_rotation_matrix)

	# 验证标定板的标定效果/精度
	src_img = rectify_image(src_img, camera_matrix, distortion_coefficients, img_shape)
	corners = get_corners(src_img, chessboard_size).reshape(-1, 2, 2)
	for point2ds in corners:
		points3d = point2world(point2ds, new_rotation_matrix, new_translation_vector.reshape(-1), camera_matrix, 0)
		# points3d = point2world(point2ds, rotation_matrixs[0], translation_vectors[0].reshape(-1), camera_matrix, 0)
		distance = point_distance(points3d)
		print(f"true distance is {distance} mm")

	# 手动寻找测量点，进行尺寸计算
	# point2ds = [[944, 2185], [4764, 2236]]

	# points3d = point2world(point2ds, new_rotation_matrix, new_translation_vector.reshape(-1), camera_matrix, -16.5)
	# # points3d = point2world(point2ds, rotation_matrixs[0], translation_vectors[0].reshape(-1), camera_matrix, 0)
	# distance = point_distance(points3d)
	# print(f"true distance is {distance} mm")
