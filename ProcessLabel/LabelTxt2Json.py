# -*- coding:utf-8 -*-
"""
@Time:2024/3/15  9:38
@Auth:yuanjinhao
@File:LabelTxt2Json.py
"""
import os
import json
from tqdm import tqdm

category_index = {'Car_front': '0', 'Car_rear': '1'}


def get_key(val, dict):
	for key, value in dict.items():
		if val == value:
			return key


def txt2json(txt_path, name, suffix, output_dir):
	txt_labels = []

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
		shape = {}
		points = [txt_label[1:3], txt_label[3:]]
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
		'imagePath': name + '.' + suffix,
		'imageWidth': 1920,
		'imageHeight': 1080}

	category_file = os.path.join(output_dir, name + '.json')
	with open(category_file, 'w') as f:
		json.dump(category_data, f, indent=4)


if __name__ == '__main__':
	images_txts_dir = './Cleaning_IntersectionData/Cleaning_20240315'

	txts_name = 'labels'
	txts_dir = os.path.join(images_txts_dir, txts_name)

	txts_name = os.listdir(txts_dir)
	images_names = os.listdir(images_txts_dir)
	for image_name in tqdm(images_names, desc='Generate', ncols=70):
		try:
			name, suffix = image_name.split('.')
			if suffix == 'jpg' or suffix == 'png':
				txt_name = name + '.txt'
				if txt_name in txts_name:
					txt_path = os.path.join(txts_dir, txt_name)
					txt2json(txt_path, name, suffix, images_txts_dir)
				else:
					image_path = os.path.join(images_txts_dir, image_name)
					os.remove(image_path)
		except:
			print('ValueError: not enough values to unpack (expected 2, got 1)')

