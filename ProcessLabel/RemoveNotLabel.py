# -*- coding:utf-8 -*-
"""
@Time:2024-10-09  15:52
@Auth:yuanjinhao
@File:RemoveNotLabel.py
"""
"""
删除没有标签的图像和标注文件
"""
import os
from tqdm import tqdm
import json
import shutil


def remove_json_not_label(json_name, json_dir, image_dir):
	image_formats = ['.jpg', '.png', '.jpeg']
	json_path = os.path.join(json_dir, json_name)
	name = os.path.splitext(json_name)[0]

	with open(json_path, 'r') as json_f:
		data = json.load(json_f)
	json_f.close()
	if data == None:
		for ext in image_formats:
			img_file = name + ext
			potential_path = os.path.join(image_dir, img_file)
			if os.path.exists(potential_path):
				os.remove(potential_path)
		os.remove(json_path)


if __name__ == "__main__":
	json_folder = '../LabeledData/PilingCarV4'
	image_folder = '../LabeledData/PilingCarV4'
	files = os.listdir(json_folder)
	for file in tqdm(files):
		if file.endswith('json'):
			remove_json_not_label(file, json_folder, image_folder)
