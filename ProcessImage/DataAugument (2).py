# -*- coding:utf-8 -*-
"""
@Time:2023/9/28  9:28
@Auth:yuanjinhao
@File:augument_02.py
"""

import os
import shutil

import albumentations as A
import cv2
from tqdm import tqdm


def get_boxes(label_file):
	categories = []
	boxes = []
	with open(label_file, 'r') as f:
		datas = f.readlines()
		for data in datas:
			data = data.strip().split(' ')
			category = data[0]
			box = list(map(float, data[1:]))
			categories.append(category)
			boxes.append(box)
		f.close()

	return categories, boxes


def augment_and_show(aug, image, bboxes, categories, out_image, out_label, lable_name, data_num):
	augmented = aug(image=image, bboxes=bboxes, category_id=categories)
	image = augmented['image']

	txt_path = os.path.join(out_label, lable_name[:-4] + data_num + '.txt')
	with open(txt_path, 'w') as f:
		for bbox, category in zip(augmented['bboxes'], augmented['category_id']):
			x_center, y_center, w, h = map(float, bbox)
			node = f'{category} {x_center} {y_center} {w} {h}\n'
			f.write(node)
		f.close()

	out_image_path = os.path.join(out_image, lable_name[:-4] + data_num + '.png')
	cv2.imwrite(out_image_path, image)
	return

# 测试绘制
# 	x_left_top = int(x_center * 1920 - w * 1920 / 2)
# 	y_left_top = int(y_center * 1080 - h * 1080 / 2)
# 	x_right_bottom = int(x_center * 1920 + w * 1920 / 2)
# 	y_right_bottom = int(y_center * 1080 + h * 1080 / 2)
#
# 	cv2.rectangle(image, (x_left_top, y_left_top), (x_right_bottom, y_right_bottom), (0, 255, 0), 2)
# 	cv2.putText(image, category, (x_left_top, y_left_top - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 0, 0), 1)
# return image


if __name__ == '__main__':
	# 定义增强变换
	transform = A.Compose([A.HorizontalFlip(p=0.7),
	                       A.ShiftScaleRotate(shift_limit=0.1, rotate_limit=0, border_mode=cv2.BORDER_CONSTANT,
	                                          value=0, p=0.5),
	                       A.OneOf([
		                       A.ISONoise(),  # 将高斯噪声添加到输入图像
		                       A.GaussNoise(),  # 将高斯噪声应用于输入图像
	                       ], p=0.5),  # 应用选定变换的概率
	                       A.OneOf([
		                       A.MotionBlur(p=0.3),  # 使用随机大小的内核将运动模糊应用于输入图像。
		                       A.MedianBlur(blur_limit=3, p=0.2),  # 中值滤波
		                       A.Blur(blur_limit=3, p=0.2),  # 使用随机大小的内核模糊输入图像。
	                       ], p=0.5),
	                       A.RandomBrightnessContrast(p=0.5),
	                       A.RGBShift(r_shift_limit=30, g_shift_limit=30, b_shift_limit=30, p=0.3),
	                       A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False,
	                                     p=0.5)],
	                      bbox_params=A.BboxParams(format='yolo', label_fields=['category_id']))

	# transform = A.Compose([A.RandomBrightnessContrast(p=1)],
	#                       bbox_params=A.BboxParams(format='yolo', label_fields=['category_id']))

	# 加载图像和标注
	labels_dir = './result/labels'
	images_dir = './result/images'
	labels_list = os.listdir(labels_dir)

	# 增强次数
	augument_time = 4

	# 输出路径
	out_dir = 'augument_data'
	if os.path.isdir(out_dir):
		shutil.rmtree(out_dir)
	os.makedirs(out_dir, exist_ok=True)
	out_labels_dir = os.path.join(out_dir, 'labels')
	out_images_dir = os.path.join(out_dir, 'images')
	os.makedirs(out_images_dir, exist_ok=True)
	os.makedirs(out_labels_dir, exist_ok=True)

	for num in range(augument_time):
		data_num = f'_{num + 1}'
		for labels in tqdm(labels_list,desc=f'augument num{num + 1}: ', ncols=80):
			labels_path = os.path.join(labels_dir, labels)
			image_path = os.path.join(images_dir, labels[:-4] + '.png')
			categories, boxes = get_boxes(labels_path)
			image = cv2.imread(image_path)
			augment_and_show(transform, image, boxes, categories, out_images_dir, out_labels_dir, labels, data_num)

# 单张测试
# file_name = '20230506-93_14100_26503.txt'
# labels_dir = 'many_front/labels'
# images_dir = 'images/train1'
# label = os.path.join(labels_dir, file_name)
# image = os.path.join(images_dir, file_name[:-4] + '.png')
# categories, boxes = get_boxes(label)
#
# img = cv2.imread(image)
# # # 执行增强变换并显示结果
# augmented_image = augment_and_show(transform, img, bboxes=boxes, categories=categories)
# cv2.imshow('Augmented', augmented_image)
# cv2.waitKey(0)
