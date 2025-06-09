# coding:utf-8
"""
Name  : track.py
Author: yuan.jinhao
Time  : 2024/10/16 下午4:00
Desc  : 追踪推理脚本，保存推理视频
"""
import os
from collections import defaultdict
import numpy as np
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


def track(opt):
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
    track_history = defaultdict(lambda: [])
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
    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            results = model.track(frame, persist=True)

            # 获取框和追踪ID
            boxes = results[0].boxes.xywh.cpu()

            annotated_frame = results[0].plot()

            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # 在帧上展示结果
                annotated_frame = results[0].plot()

                # 绘制追踪路径
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))  # x, y中心点
                    if len(track) > 30:  # 在90帧中保留90个追踪点
                        track.pop(0)

                    # 绘制追踪线
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 展示带注释的帧
            out.write(annotated_frame)
            if show:
                # 展示带注释的帧
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
            # 如果按下'q'则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果视频结束则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    opt = parse_opt()
    track(opt)
