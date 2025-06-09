# coding:utf-8
"""
Name  : ReadBinaryFile.py
Author: yuan.jinhao
Time  : 2024/10/24 下午6:23
Desc  :
"""
import struct


record_format = 'f'
record_size = struct.calcsize(record_format)
# 打开二进制文件
with open('./Files/100.iccal', 'rb') as file:
    while True:

        record_data = file.read(record_size)
        if not record_data:
            break  # 如果没有更多数据，则退出循环

        # 解包记录数据
        record = struct.unpack(record_format, record_data)
        print(record)

