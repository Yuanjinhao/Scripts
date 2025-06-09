# -*- coding:utf-8 -*-
"""
@Time:2024-10-14  14:57
@Auth:yuanjinhao
@File:RenameFiles.py
"""
import os
from tqdm import tqdm
import json

"""
将中文文件名转换为英文文件名
"""


def rename_file(old_name, folder, id, suffix, image_suffix=""):
	dir_name = os.path.basename(folder)
	fixed_length_str = "{:06}".format(id)
	new_name = dir_name + '_' + fixed_length_str + suffix

	old_path = os.path.join(folder, old_name)
	new_path = os.path.join(folder, new_name)
	os.rename(old_path, new_path)

	if suffix == '.json':
		with open(new_path, 'r', encoding='utf-8') as json_f:
			data = json.load(json_f)
		if data == None:
			return
		image_name = dir_name + '_' + fixed_length_str + image_suffix
		category_data = {'version': data['version'],
		                 'flags': data['flags'],
		                 'shapes': data['shapes'],
		                 'imagePath': image_name,
		                 'imageData': 'null',
		                 'imageHeight': data['imageHeight'],
		                 'imageWidth': data['imageWidth']}
		with open(new_path, 'w', encoding='utf-8') as file:
			json.dump(category_data, file, indent=4)


if __name__ == "__main__":
	image_formats = ['.jpg', '.png', '.jpeg']
	files_folder = "../LabeledData/PilingCarV8"
	files_name = os.listdir(files_folder)
	id = 1
	for file_name in tqdm(files_name):
		name, json_suffix = os.path.splitext(file_name)
		if json_suffix == ".json":
			image_name = ""
			image_suffix = ""
			for ext in image_formats:
				img_file = name + ext
				potential_path = os.path.join(files_folder, img_file)
				if os.path.exists(potential_path):
					image_name = img_file
					image_suffix = ext
					break

			rename_file(file_name, files_folder, id, json_suffix, image_suffix)
			rename_file(image_name, files_folder, id, image_suffix)
			id += 1
