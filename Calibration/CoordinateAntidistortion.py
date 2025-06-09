# -*- coding:utf-8 -*-
"""
@Time:2024/6/5  9:21
@Auth:yuanjinhao
@File:coordinate_antidistortion.py
"""
"""
图像反畸变，获取测量点的在原图上的坐标值
"""

from find_measurement_points import *

if __name__ == "__main__":
    # 内参矩阵
    camera_matrix = np.array([[1.47398380e+04, 0.00000000e+00, 2.69164907e+03],
                              [0.00000000e+00, 1.47392043e+04, 1.87747596e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    distortion_coefficients = np.array([-6.44795502e-02, 1.26520250e+00, 8.86868577e-05,
                                        6.65685463e-04, 7.55022133e+01])

    # 重标定的外参矩阵
    translation_vector = np.array([38.12476686, -31.7455947, 753.10446671])
    rotation_matrix = np.array([[0.01284695, -0.99927522, 0.03583275],
                                 [0.99979552, 0.01339681, 0.01514749],
                                 [-0.01561656, 0.03563082, 0.999243]])

    image_path = './images/testimages0603/37.bmp'
    src_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_shape = src_img.shape
    measuring_area = [[137 * 4, 360 * 4], [1298 * 4, 663 * 4]]

    rec_img = rectify_image(src_img, camera_matrix, distortion_coefficients, img_shape)

    point2ds = find_measurement_points(rec_img, measuring_area, img_shape, use_reference='top')

    # 将矫正点坐标映射回畸变的原图坐标
    new_point2ds = distor_points1(points=point2ds, camera_matrix=camera_matrix,
                                  distortion_coeffs=distortion_coefficients, image_shape=img_shape)
    new_point2ds = np.array(new_point2ds, dtype=np.int32)
    cv2.line(src_img, new_point2ds[0], new_point2ds[1], (255, 255, 255), 2)
    show("src img", src_img)

    cv2.line(rec_img, point2ds[0], point2ds[1], (255, 255, 255), 2)
    show("rec img", rec_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    points3d = point2world(point2ds, rotation_matrix, translation_vector, camera_matrix, -16.5)
    # points3d = point2world(point2ds, rotation_matrixs[0], translation_vectors[0].reshape(-1), camera_matrix, 0)
    distance = point_distance(points3d)
    print(f"true distance is {distance} mm")
