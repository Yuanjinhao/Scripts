# coding:utf-8
"""
Name  : PreLabel.py
Author: yuan.jinhao
Time  : 2024/11/4 上午9:06
Desc  : 预标注脚本，去除重复的图像，并对图像进行预标注
"""

import os
import argparse
import shutil
import json
from pathlib import Path
from tqdm import tqdm
from natsort import natsorted
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
category_index = {'0': 'forklift_legal', '1': 'forklift_illegal', '2': 'forklift_legal_vague',
                  '3': 'forklift_illegal_vague'}


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./weights/yolov8n.pt", help="initial weights path")
    parser.add_argument("--input", type=str, default=ROOT / "./datasets/coco8/images/val", help="input video path")
    parser.add_argument("--output", type=str, default=None, help="output dir path")
    parser.add_argument("--iou", type=float, default=0.3, help="Remove images with IOU greater than the threshold")
    parser.add_argument("--prelabel", type=bool, default=True, help="Whether is not to prelabel")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def box_iou(box_1, box_2):
    x1 = box_1[0] if box_1[0] > box_2[0] else box_2[0]
    y1 = box_1[1] if box_1[1] > box_2[1] else box_2[1]
    x2 = box_1[2] if box_1[2] < box_2[2] else box_2[2]
    y2 = box_1[3] if box_1[3] < box_2[3] else box_2[3]

    w = x2 - x1 if x2 - x1 > 0 else 0
    h = y2 - y1 if y2 - y1 > 0 else 0

    inter_area = w * h
    union_area = ((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]) +
                  (box_2[2] - box_2[0]) * (box_2[3] - box_2[1]) - inter_area)

    return inter_area / union_area


def save2json(result, image_name, image_shape, output_path):
    shapes = []
    # 根据iou进行判断
    for box in result.boxes:
        box0 = box.xyxy.cpu().tolist()[0]
        cls = str(int(box.cls.cpu()[0]))
        shape = {}
        points = [[box0[0], box0[1]], [box0[2], box0[3]]]
        category = category_index.get(cls, 'none')

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
        'imagePath': image_name,
        'imageWidth': image_shape[1],
        'imageHeight': image_shape[0]}

    with open(output_path, 'w') as f:
        json.dump(category_data, f, indent=4)


def preLabel(opt):
    weights = opt.weights
    input_dir = opt.input
    output_dir = opt.output
    iou_threshold = opt.iou
    prelabel = opt.prelabel
    if output_dir == None:
        output_dir = input_dir

    # 加载模型
    model = YOLO(weights)

    # 图像路径
    images_name = natsorted(os.listdir(input_dir))

    # 创建保存文件夹
    output_images_path = os.path.join(output_dir, 'images')
    output_labels_path = os.path.join(output_dir, 'labels')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_images_path, exist_ok=True)
    os.makedirs(output_labels_path, exist_ok=True)

    temp_result = None
    for image_name in tqdm(images_name):
        # 是否删除标志
        flag = False
        image_path = os.path.join(input_dir, image_name)

        # 运行推理
        result = model.predict(image_path, save=False)[0]

        # 没有目标的直接删除
        if len(result.boxes.cls) == 0:
            os.remove(image_path)
            continue

        # 初始化暂存的结果
        if temp_result is None:
            temp_result = result

        else:
            # 匹配当前图像与之前保留的图像的IOU，并去除重合度高的图像
            for box1 in result.boxes:
                box_1 = box1.xyxy.cpu().tolist()[0]

                for box2 in temp_result.boxes:
                    box_2 = box2.xyxy.cpu().tolist()[0]
                    iou = box_iou(box_1, box_2)
                    if iou > iou_threshold:
                        flag = True

        if flag:
            # 删除重复高的的图像
            os.remove(image_path)

        else:
            if prelabel:
                # 移动图像路径
                new_image_path = os.path.join(output_images_path, image_name)
                shutil.move(image_path, new_image_path)

                # 创建标注文件
                new_label_path = os.path.join(output_labels_path, image_name.split('.')[0] + '.json')
                save2json(result, image_name, result.orig_shape[:2], new_label_path)

            temp_result = result


if __name__ == "__main__":
    opt = parse_opt()
    preLabel(opt)
