# -*- coding:utf-8 -*-
"""
@Time:2023/12/18  15:35
@Auth:yuanjinhao
@File:Sampling_file.py
"""
import os
import random
import shutil
from tqdm import tqdm

if __name__ == '__main__':
	src_dir = './TFLDatasets'
	det_dir = './IntersectionData'

	os.makedirs(det_dir, exist_ok=True)
	count = 0

	src_files = os.listdir(src_dir)
	for src_file in tqdm(random.sample(src_files, 2000)):
		name, suffix = src_file.split('.')
		# 有标注的文件
		# if suffix == 'json':
		# 	src_json = os.path.join(src_dir, src_file)
		# 	det_json = os.path.join(det_dir, src_file)
		#
		# 	img = name + '.png'
		# 	src_img = os.path.join(src_dir, img)
		# 	det_img = os.path.join(det_dir, img)
		#
		# 	shutil.move(src_json, det_json)
		# 	shutil.move(src_img, det_img)
		#
		# 	count += 1

		# 只有图像
		if suffix == 'png' or suffix == 'jpg':
			src_json = os.path.join(src_dir, src_file)
			det_json = os.path.join(det_dir, src_file)

			shutil.move(src_json, det_json)

			count += 1

		if count == 2000:
			break

