# -*- coding:utf-8 -*-
"""
@Time:2023/12/18  12:13
@Auth:yuanjinhao
@File:Modify_small_target_labels.py
"""

import os
import shutil
import json
from tqdm import tqdm


def modify_small_target(src_path, det_path):
	try:

		with open(src_path, 'r') as f:
			datas = json.load(f)
			shapes = []
			for data in datas['shapes']:
				shape = {}
				points = data['points']
				left_top = points[0]
				right_buttom = points[1]
				if (right_buttom[0] - left_top[0]) < 20 or (right_buttom[1] - left_top[1]) < 20:
					left_top = [x + 1 for x in left_top]
					right_buttom = [x - 1 for x in right_buttom]
					points = [left_top, right_buttom]

				shape['flags'] = []
				shape['label'] = data['label']
				shape['points'] = points
				shape['group_id'] = None
				shape['shape_type'] = "rectangle"
				shapes.append(shape)

		f.close()
		category_data = {
			'flags': {},
			'shapes': shapes,
			'version': "5.0.1",
			'imageData': None,
			'imagePath': src_label.split('\\')[-1][:-4] + 'png',
			'imageWidth': 1920,
			'imageHeight': 1080
		}

		with open(det_path, 'w') as f:
			json.dump(category_data, f, indent=4)

		return True

	except:
		print(f"Error: {src_label} No Label!")
		return False


if __name__ == '__main__':
	src_labels_dir = "./TDA4FrontRear20231208"
	det_labels_dir = "./TDA4FrontRear20231208_m"
	os.makedirs(det_labels_dir, exist_ok=True)

	src_labels = os.listdir(src_labels_dir)
	for src_label in tqdm(src_labels):
		name, suffix = src_label.split('.')
		if suffix == 'json':
			src_path = os.path.join(src_labels_dir, src_label)
			det_path = os.path.join(det_labels_dir, src_label)

			if modify_small_target(src_path, det_path):
				img_name = name + '.png'
				src_img = os.path.join(src_labels_dir, img_name)
				det_img = os.path.join(det_labels_dir, img_name)

				shutil.copy(src_img, det_img)
