# coding:utf-8
"""
Name  : GetMisidentification.py
Author: yuan.jinhao
Time  : 2024/10/21 下午4:07
Desc  : 推理过程中，将误判的图像单独保留
"""

import os
import argparse
import cv2
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
category_index = {'0': 'forklift_legal', '1': 'forklift_illegal'}


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./weights/yolov8n.pt", help="initial weights path")
    parser.add_argument("--input", type=str, default=ROOT / "./datasets/coco8/images/val", help="input video path")
    parser.add_argument("--output", type=str, default="./runs/mistaked", help="output video path")
    parser.add_argument("--save", type=bool, default=True, help="whether is not to save result")

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


def read_txt(txt_path, width, height):
    labels_list = []
    try:
        with open(txt_path, 'r') as f:
            datas = f.readlines()
            for data in datas:
                data = data.split(' ')
                category = category_index.get(data[0], 'none')
                x_center, y_center, w, h = list(map(float, data[1:5]))
                x_left_top = float(x_center * width - w * width / 2)
                y_left_top = float(y_center * height - h * height / 2)
                x_right_bottom = float(x_center * width + w * width / 2)
                y_right_bottom = float(y_center * height + h * height / 2)
                labels_list.append([category, x_left_top, y_left_top, x_right_bottom, y_right_bottom])
        f.close()
        return labels_list
    except:
        print(f"Error: {txt_path} not label information! ")

    return


def inference(opt):
    weights = opt.weights
    input_dir = opt.input
    output_dir = opt.output
    save = opt.save

    # 加载模型
    model = YOLO(weights)  # 根据需要选择模型权重

    # 图像路径
    images_name = os.listdir(input_dir)

    # 创建保存文件夹
    os.makedirs(output_dir, exist_ok=True)

    for image_name in tqdm(images_name):
        image_path = os.path.join(input_dir, image_name)

        # 获取标签文件
        sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"  # /images/, /labels/ substrings
        label_path = sb.join(image_path.rsplit(sa, 1)).rsplit(".", 1)[0] + ".txt"

        # 运行推理
        result = model.predict(image_path, save=False)[0]
        img = result.orig_img  # 获取原始图像
        targets_list = read_txt(label_path, result.orig_shape[1], result.orig_shape[0])
        misclassified = save

        # 匹配预测和真实框，并保留类别不相等的图像
        for box in result.boxes:
            box_1 = box.xyxy.cpu().tolist()[0]
            cls = str(int(box.cls.cpu()[0]))
            for i, target in enumerate(targets_list):
                box_2 = target[1:]
                iou = box_iou(box_1, box_2)
                if iou > 0.5 and category_index.get(cls, 'none') != target[0]:
                    misclassified = True
                    break

        if misclassified:
            # 绘制预测矩形框
            for box in result.boxes:
                box0 = box.xyxy.cpu().tolist()[0]
                cls = str(int(box.cls.cpu()[0]))
                point_1 = [int(box0[0]), int(box0[1])]
                point_2 = [int(box0[2]), int(box0[3])]
                cv2.rectangle(img, point_1, point_2, color=(0, 0, 255), thickness=5)
                cv2.putText(img, category_index.get(cls, 'none'), point_1, color=(0, 0, 255), thickness=2, fontScale=2,
                            fontFace=2)
            # 绘制真实矩形框
            for box in targets_list:
                point_1 = [int(box[1]), int(box[2])]
                point_2 = [int(box[3]), int(box[4])]
                cv2.rectangle(img, point_1, point_2, color=(0, 255, 0), thickness=5)
                cv2.putText(img, box[0], point_1, color=(0, 255, 0), thickness=2, fontScale=2, fontFace=2)

            save_path = output_dir + '/' + image_name
            cv2.imwrite(save_path, img)  # 保存误识别的图像


if __name__ == "__main__":
    opt = parse_opt()
    inference(opt)
