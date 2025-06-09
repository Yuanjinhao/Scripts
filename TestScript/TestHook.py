# coding:utf-8
"""
Name  : TestHook.py
Author: yuan.jinhao
Time  : 2024/10/15 下午2:26
Desc  : 钩子函数功能测试
"""

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = MyModel()

activations = []

# 钩子函数：用于捕获激活值
def save_activation(module, input, output):
    activations.append(output)

# 注册钩子
handle = model.conv1.register_forward_hook(save_activation)

# 进行一次前向传播
input_data = torch.randn(1, 1, 28, 28)  # 随机输入数据
output = model(input_data)

# 现在 activations 列表中保存了 conv1 层的输出
print(activations[0].shape)

# 移除钩子
handle.remove()
