# -*- coding:utf-8 -*-
"""
@Time:2023/11/30  12:02
@Auth:yuanjinhao
@File:merge_labels.py
"""

import os
import json
import shutil
from tqdm import tqdm

category_index = {'Car_front': '0', 'Car_rear': '1'}


def get_key(val, dict):
	for key, value in dict.items():
		if val == value:
			return key


def box_iou(box_1, box_2):
	x1 = box_1[0] if box_1[0] > box_2[0] else box_2[0]
	y1 = box_1[1] if box_1[1] > box_2[1] else box_2[1]
	x2 = box_1[2] if box_1[2] < box_2[2] else box_2[2]
	y2 = box_1[3] if box_1[3] < box_2[3] else box_2[3]

	w = x2 - x1 if x2 - x1 > 0 else 0
	h = y2 - y1 if y2 - y1 > 0 else 0

	over_area = w * h
	return over_area / ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) +
	                    (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]) - over_area)


def merge_label(json_path, txt_path, json_name, output_dir):
	json_labels = []
	txt_labels = []

	try:
		with open(json_path, 'r') as f:
			datas = json.load(f)
			for data in datas['shapes']:
				label = data['label']
				if ('Car_front' in label) or ('Car_rear' in label):
					json_labels.append([label, data['points'][0][0], data['points'][0][1],
					                    data['points'][1][0], data['points'][1][1]])
		f.close()
	except:
		print(f"Error: {json_path} not label information!")

	try:
		with open(txt_path, 'r') as f:
			datas = f.readlines()
			for data in datas:
				data = data.split(' ')
				category = get_key(data[0], category_index)
				x_center, y_center, w, h = list(map(float, data[1:5]))
				x_left_top = float(x_center * 1920 - w * 1920 / 2)
				y_left_top = float(y_center * 1080 - h * 1080 / 2)
				x_right_bottom = float(x_center * 1920 + w * 1920 / 2)
				y_right_bottom = float(y_center * 1080 + h * 1080 / 2)
				txt_labels.append([category, x_left_top, y_left_top, x_right_bottom, y_right_bottom])
		f.close()
	except:
		print(f"Error: {txt_path} not label information! ")

	shapes = []
	# 根据iou进行判断
	for i, txt_label in enumerate(txt_labels):
		iou_max = 0
		temp_index = 0
		shape = {}
		points = [txt_label[1:3], txt_label[3:]]
		for j, json_label in enumerate(json_labels):
			box_1 = txt_label[1:]
			box_2 = json_label[1:]
			iou = box_iou(box_1, box_2)
			if iou > iou_max:
				iou_max = iou
				temp_index = j

		if iou_max > 0.5:
			category = json_labels[temp_index][0]
		else:
			category = txt_label[0]

		shape['flags'] = []
		shape['label'] = category
		shape['points'] = points
		shape['group_id'] = None
		shape['shape_type'] = "rectangle"
		shapes.append(shape)

	category_data = {
		'flags': {},
		'shapes': shapes,
		'version': "5.0.1",
		'imageData': None,
		'imagePath': json_path.split('\\')[-1][:-4] + 'png',
		'imageWidth': 1920,
		'imageHeight': 1080

	}

	category_file = os.path.join(output_dir, json_name)
	with open(category_file, 'w') as f:
		json.dump(category_data, f, indent=4)


def copy_image(image, old_dir, new_dir):
	old_path = os.path.join(old_dir, image)
	new_path = os.path.join(new_dir, image)

	shutil.copy(old_path, new_path)


if __name__ == '__main__':
	images_jsons_dir = 'Data_20231018(remain)'
	txts_dir = 'labels'

	output_dir = images_jsons_dir + '_merge'
	os.makedirs(output_dir, exist_ok=True)

	names = os.listdir(txts_dir)
	for txt_name in tqdm(names, desc='merge', ncols=70):
		name, suffix = txt_name.split('.')

		json_name = name + '.json'
		image_name = name + '.png'

		json_path = os.path.join(images_jsons_dir, json_name)
		txt_path = os.path.join(txts_dir, txt_name)
		if os.path.exists(txt_path):
			merge_label(json_path, txt_path, json_name, output_dir)
			copy_image(image_name, images_jsons_dir, output_dir)
