# -*- coding:utf-8 -*-
"""
@Time:2023/9/8  8:48
@Auth:yuanjinhao
@File:draw_json_rectangle.py
"""
import json
import os
from tqdm import tqdm
import numpy as np
import cv2
import json


def draw_rectangle(image, image_path, label_path, out_dir):
	img = cv2.imread(image_path)
	point = np.array([[0, 1079], [0, 650], [930, 500], [990, 500], [1919, 650], [1919, 1079], [0, 1079]])
	img = cv2.polylines(img, [point], False, (0, 0, 255), 1)

	try:
		with open(label_path, 'r') as f:
			datas = json.load(f)
			for data in datas['shapes']:
				label = data['label']
				if 'Car_front' in label or 'Car_rear' in label:
					points = data['points']
					point_left_top = list(map(int, points[0]))
					point_right_bottom = list(map(int, points[1]))
					# 计算面积
					label += f"--width:{point_right_bottom[0] - point_left_top[0]}, " \
					         f"height: {(point_right_bottom[1] - point_left_top[1])} "
					cv2.putText(img, label, (point_left_top[0], point_left_top[1] - 5), cv2.FONT_HERSHEY_PLAIN, 0.8,
					            (255, 0, 0), 1)
					cv2.rectangle(img, point_left_top, point_right_bottom, (255, 0, 0), 2)

		file_name = os.path.join(out_dir, image)
		cv2.imwrite(file_name, img)
	except:
		print(f"Error: {label_path}")


if __name__ == "__main__":
	# 绘制一步清洗后的图像
	dir_path = "Cleaning_TDA4FrontRear20231222/TDA4FrontRear20231228"
	images_list = os.listdir(dir_path)
	out_dir = os.path.join(dir_path, 'draw_images')
	os.makedirs(out_dir, exist_ok=True)
	for image in tqdm(images_list[:500], desc="draw labels", ncols=70, position=0, leave=True):
		if image[-4:] == '.png':
			image_path = os.path.join(dir_path, image)
			label_path = os.path.join(dir_path, image[:-4] + '.json')
			draw_rectangle(image, image_path, label_path, out_dir)
