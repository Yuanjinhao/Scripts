# -*- coding:utf-8 -*-
"""
@Time:2024/3/15  10:16
@Auth:yuanjinhao
@File:RemoveSingle.py
"""
import os
from tqdm import tqdm

input_dir = './IntersectionData'
files_list = os.listdir(input_dir)

for file in tqdm(files_list, desc='Remove', ncols=70):
	name, suffix = file.split('.')
	if suffix == 'jpg' or suffix == 'png':
		json_name = name + '.json'
		if json_name in files_list:
			pass
		else:
			image_path = os.path.join(input_dir, file)
			os.remove(image_path)




