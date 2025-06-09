# -*- coding:utf-8 -*-
"""
@Time:2024/1/23  13:52
@Auth:yuanjinhao
@File:Count_Front&Rear_num.py
"""
# 此方法为计算数据集中有多少车头车尾，并计算大、中、小实例数量
import os
from tqdm import tqdm


# 读取label文件
def read_label(label_path, small_thres, big_thres):
	front_num = 0
	rear_num = 0
	big_num = 0
	med_num = 0
	small_num = 0

	with open(label_path, 'r') as f:
		datas = f.readlines()
		for data in datas:
			# 类别计数
			data = data.strip().split(' ')
			if data[0] == '0':
				front_num += 1
			elif data[0] == '1':
				rear_num += 1

			pos = list(map(float, data[1:]))
			x_min = (pos[0] - pos[2] / 2) * 1920
			y_min = (pos[1] - pos[3] / 2) * 1080
			x_max = (pos[0] + pos[2] / 2) * 1920
			y_max = (pos[1] + pos[3] / 2) * 1080
			if (x_max - x_min) < small_thres or (y_max - y_min) < small_thres:
				small_num += 1
			elif small_thres <= (x_max - x_min) < big_thres or small_thres <= (y_max - y_min) < big_thres:
				med_num += 1
			else:
				big_num += 1

	f.close()
	return front_num, rear_num, small_num, med_num, big_num


if __name__ == '__main__':
	# 阈值设置
	small_thres = 36
	big_thres = 96

	Front_num = 0
	Rear_num = 0
	Small_num = 0
	Med_num = 0
	Big_num = 0

	labels_dir = './labels'
	for label_root, label_dirs, label_files in os.walk(labels_dir):
		if 'train_20' in label_root or 'train_TDA4' in label_root:
			for label_name in tqdm(label_files, desc=label_root + ': ', ncols=80):
				label_path = os.path.join(label_root, label_name)
				front_num, rear_num, small_num, med_num, big_num = read_label(label_path, small_thres, big_thres)
				Front_num += front_num
				Rear_num += rear_num
				Small_num += small_num
				Med_num += med_num
				Big_num += big_num

	print(f"Front num:{Front_num}, Rear num:{Rear_num}, Small num:{Small_num}, Med num:{Med_num}, Big num:{Big_num}")
