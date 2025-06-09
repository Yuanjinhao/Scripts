# -*- coding:utf-8 -*-
"""
@Time:2023/9/7  10:51
@Auth:yuanjinhao
@File:draw_txt_rectangle.py
"""
import os
from tqdm import tqdm
import numpy as np
import cv2


# 在图像上绘制标注框
def draw_rectangle(image, image_path, label_path, out_dir):
	# category_index = {'Car_front': '0', 'Car_rear': '1'}

	# category_index = {'Car_front': '0', 'Car_front_left': '1', 'Car_front_right': '2', 'Car_front_lr': '3',
	#                   'Car_rear': '4', 'Car_rear_left': '5', 'Car_rear_right': '6', 'Car_rear_lr': '7',
	#                   'Car_rear_stoplight': '8', 'Car_rear_stoplight_left': '9', 'Car_rear_stoplight_right': '10',
	#                   'Car_rear_stoplight_lr': '11'}

	# category_index = { 'Car_front_left': '0', 'Car_front_right': '1', 'Car_front_lr': '2',
	#                    'Car_rear_left': '3', 'Car_rear_right': '4', 'Car_rear_lr': '5',
	#                   'Car_rear_stoplight': '6', 'Car_rear_stoplight_left': '7', 'Car_rear_stoplight_right': '8',
	#                   'Car_rear_stoplight_lr': '9'}

	category_index = {'Car_front': '0', 'Car_headlight_left': '1', 'Car_headlight_right': '2', 'Car_headlight_lr': '3',
	                  'Car_headlight': '4', 'Car_front_left': '5', 'Car_front_right': '6', 'Car_front_lr': '7','Car_rear': '8',
	                  'Car_rear_left': '9', 'Car_rear_right': '10', 'Car_rear_lr': '11','Car_rear_stoplight': '12',
	                  'Car_rear_stoplight_left': '13', 'Car_rear_stoplight_right': '14','Car_rear_stoplight_lr': '15'}


	img = cv2.imread(image_path)
	point = np.array([[0, 1079], [0, 650], [930, 500], [990, 500], [1919, 650], [1919, 1079], [0, 1079]])
	img = cv2.polylines(img, [point], False, (0, 0, 255), 1)
	with open(label_path, 'r') as f:
		datas = f.readlines()
		for data in datas:
			data = data.split(' ')
			category = get_key(data[0], category_index)
			x_center, y_center, w, h = list(map(float, data[1:5]))
			x_left_top = int(x_center * 1920 - w * 1920 / 2)
			y_left_top = int(y_center * 1080 - h * 1080 / 2)
			x_right_bottom = int(x_center * 1920 + w * 1920 / 2)
			y_right_bottom = int(y_center * 1080 + h * 1080 / 2)
			cv2.putText(img, category, (x_left_top, y_left_top - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
			cv2.rectangle(img, (x_left_top, y_left_top), (x_right_bottom, y_right_bottom), (255, 0, 0), 2)

	file_name = os.path.join(out_dir, image)
	cv2.imwrite(file_name, img)


def get_key(val, dict):
	for key, value in dict.items():
		if val == value:
			return key


if __name__ == "__main__":
	# 绘制经过两步清洗后的图像
	dir_path = "LightAttributeDatasets/Cleaning_TDA4FrontRear20231127/regionOverlapSize/Data"
	images_path = os.path.join(dir_path, 'images/train_TDA4FrontRear20231127')
	images_list = os.listdir(images_path)
	out_dir = os.path.join(dir_path, 'draws')
	os.makedirs(out_dir, exist_ok=True)
	for image in tqdm(images_list[:100]):
		image_path = os.path.join(images_path, image)
		label_path = os.path.join(os.path.join(dir_path, 'labels/train_TDA4FrontRear20231127'), image[:-4] + '.txt')
		draw_rectangle(image, image_path, label_path, out_dir)
