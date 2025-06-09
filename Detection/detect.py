# coding:utf-8
"""
Name  : detect.py
Author: yuan.jinhao
Time  : 2024/10/16 下午4:01
Desc  : 检测脚本，可保存推理视频
"""
import os
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./weights/yolov8n.pt", help="initial weights path")
    parser.add_argument("--input", type=str, default=ROOT / "./data/20240105.mp4", help="input video path")
    parser.add_argument("--output_name", type=str, help="output video path")
    parser.add_argument("--show", type=bool, default=False, help="Whether or not to display images")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def detect(opt):
    # 获取参数
    weights = opt.weights
    video_path = opt.input
    output_name = opt.output_name
    show = opt.show

    # 加载模型
    model = YOLO(weights)
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 存储追踪历史
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 输出路径
    if output_name is None:
        output_name = os.path.basename(video_path)
    output_dir = "./runs/results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, output_name)

    # 构建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (frame_width, frame_height))
    # 循环视频帧
    while cap.isOpened():
        # 读取某一帧
        success, frame = cap.read()
        if success:
            # 使用yolov8进行预测
            results = model(frame)
            # 可视化结果
            annotated_frame = results[0].plot()
            # 将带注释的帧写入视频文件
            out.write(annotated_frame)
            if show:
                # 展示带注释的帧
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # 如果按下'q'则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 最后结尾中断视频帧循环
            break
    # 释放读取和写入对象
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_opt()
    detect(opt)
