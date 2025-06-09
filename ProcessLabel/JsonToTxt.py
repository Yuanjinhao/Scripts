# -*- coding:utf-8 -*-
"""
@Time:2024-09-25  15:17
@Auth:yuanjinhao
@File:JsonToTxt.py
"""
import os
import json
import cv2
import numpy as np
from PIL import Image
import shutil
from tqdm import tqdm


class Json2Txt:
	"""
	读取json标注信息转为txt标注，并生成yolo所需训练结构
	"""

	def __init__(self, json_folder, image_folder, output_folder, labels, split_ratio):
		self.json_folder = json_folder
		self.image_folder = image_folder
		self.output_folder = output_folder
		self.labels = labels
		self.split_ratio = split_ratio

		self.image_formats = ['.jpg', '.png', '.jpeg']
		self.json_list = []
		self.images_list = []
		self.legal_num = 0
		self.illegal_num = 0
		self.images_train_folder = ''
		self.images_val_folder = ''
		self.labels_train_folder = ''
		self.labels_val_folder = ''

	def get_josns(self):
		files = os.listdir(self.json_folder)
		for file in files:
			if file.endswith('json'):
				self.json_list.append(file)

	def get_images(self):
		files = os.listdir(self.image_folder)
		for file in files:
			if file.endswith(('.jpg', '.jpeg', '.png')):
				self.images_list.append(file)

	def json2txt(self, json_name, output_folder):
		txt_name = json_name.split('.')[0] + '.txt'
		txt_path = os.path.join(output_folder, txt_name)
		if os.path.exists(txt_path):
			os.remove(txt_path)
			print(f"json2txt: 删除已经存在txt，避免追加重复标签 {txt_path}")
		with open(os.path.join(self.json_folder, json_name), 'r', encoding='utf-8') as json_f:
			data = json.load(json_f)
		json_f.close()
		if data == None:
			return False
		# 读取图像尺寸
		try:
			img = cv2.imread(os.path.join(self.image_folder, data['imagePath']))
			height, width, _ = img.shape
		except:
			print(f"Picture ERROR: {json_name.split('.')[0]} cant open!")
		# 处理每个物体
		flag = False
		for shape in data['shapes']:
			# 处理越界的情况
			points = np.array(shape['points'])
			points[points < 0] = 0
			points[points >= width] = width - 1
			if points[1][1] >= height:
				points[1][1] = height - 1
			shape['points'] = list(points)
			if shape['label'] in self.labels:
				label = shape['label']
				if label == 'forklift_legal':
					self.legal_num += 1
				if label == 'forklift_illegal':
					self.illegal_num += 1

				class_index = self.labels.index(label)
				t_x, t_y = np.array(shape['points'][0])
				d_x, d_y = np.array(shape['points'][1])
				x_center = ((t_x + d_x) / 2) / width
				y_center = ((t_y + d_y) / 2) / height
				w_norm = (d_x - t_x) / width
				h_norm = (d_y - t_y) / height
				# 保存到txt文件中
				with open(txt_path, 'a') as txt_f:
					txt_f.write(f"{class_index} {x_center} {y_center} {w_norm} {h_norm}\n")
				txt_f.close()
				flag = True
		if flag:
			return True
		else:
			return False

	def make_folder(self):
		os.makedirs(self.output_folder, exist_ok=True)
		images_folder = self.output_folder + '/images'
		labels_folder = self.output_folder + '/labels'

		input_folder_name = self.json_folder.split('/')[-1]
		self.images_train_folder = images_folder + '/train' + '_' + input_folder_name
		self.images_val_folder = images_folder + '/val'
		self.labels_train_folder = labels_folder + '/train' + '_' + input_folder_name
		self.labels_val_folder = labels_folder + '/val'

		os.makedirs(self.images_train_folder, exist_ok=True)
		os.makedirs(self.images_val_folder, exist_ok=True)
		os.makedirs(self.labels_train_folder, exist_ok=True)
		os.makedirs(self.labels_val_folder, exist_ok=True)

	def run(self):
		self.get_josns()
		self.get_images()
		self.make_folder()

		i = 0
		for json_file in tqdm(self.json_list):
			if (self.split_ratio > 0) and (i % (1 / self.split_ratio) == 1):
				dst_txt_folder = self.labels_val_folder
				dst_img_folder = self.images_val_folder
			else:
				dst_txt_folder = self.labels_train_folder
				dst_img_folder = self.images_train_folder

			# 判断图像是否存在
			json_name = os.path.splitext(json_file)[0]
			for ext in self.image_formats:
				img_file = json_name + ext
				potential_path = os.path.join(self.image_folder, img_file)
				if os.path.exists(potential_path):
					src_img_path = potential_path
					break
			flag = self.json2txt(json_file, dst_txt_folder)
			if flag:
				if src_img_path.endswith("jpeg"):
					dst_img_path = os.path.join(dst_img_folder, json_name + ".jpg")
					img = Image.open(src_img_path)
					img.save(dst_img_path)  # 保存为新文件
				else:
					dst_img_path = os.path.join(dst_img_folder, img_file)
					shutil.copy(src_img_path, dst_img_path)
				i += 1


if __name__ == "__main__":
	json_folder = '../LabeledData/PilingCarV8'
	image_folder = '../LabeledData/PilingCarV8'
	outpur_folder = './PilingCar'
	# PilingCar的标签
	labels = ['forklift_legal', 'forklift_illegal', 'forklift_vague_legal', 'forklift_vague_illegal']
	# PilingCar2的标签
	# labels = ['forklift_legal', 'forklift_illegal']
	split_ratio = 0.1

	json2txt = Json2Txt(json_folder, image_folder, outpur_folder, labels, split_ratio)
	json2txt.run()
