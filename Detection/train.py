# coding:utf-8
"""
Name  : train.py
Author: yuan.jinhao
Time  : 2024/10/10 下午2:46
Desc  : 模型训练脚本，可设置一些常用的超参数
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "./weights/yolov8n.pt", help="initial weights path")
    parser.add_argument("--data", type=str, default="coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch", type=int, default=32, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--close_mosaic", type=int, default=10, help="close mosaic in last n epochs")

    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--name", default="exp", help="save to project/name")

    return parser.parse_known_args()[0] if known else parser.parse_args()


def train(opt):
    # 获取参数
    weights = opt.weights
    data = opt.data
    epochs = opt.epochs
    imgsz = opt.imgsz
    device = opt.device
    batch = opt.batch
    close_mosaic = opt.close_mosaic
    name = opt.name

    # Load a COCO-pretrained YOLO11n model
    model = YOLO(weights)

    # Train the model on the COCO8 example dataset for 100 epochs
    model.train(data=data, epochs=epochs, imgsz=imgsz, device=device, batch=batch, name=name, close_mosaic=close_mosaic)


if __name__ == "__main__":
    opt = parse_opt()
    train(opt)
