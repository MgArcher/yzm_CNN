# encoding: utf-8
"""
!/usr/bin/python3
@File: CTC_data.py
@Author:jiajia
@time: 2019/11/6 16:42
"""
import numpy as np
from keras.utils import Sequence

from CTC.settings import *
from CTC.image_data import get_image


# 创建训练集
class CaptchaSequence(Sequence):
    def __init__(self, batch_size=BATCH_SIZE, steps=STEPS):
        self.batch_size = batch_size
        self.steps = steps

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        # 定义输入输出矩阵的形状
        X = np.zeros((self.batch_size, HEIGHT, WIDTH, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, N_LEN), dtype=np.uint8)
        input_length = np.ones(self.batch_size) * INPUT_LENGHT
        label_length = np.ones(self.batch_size) * N_LEN
        img_generator = get_image()
        for i in range(self.batch_size):
            image, random_str = img_generator.__next__()
            # 图片转换成矩阵
            X[i] = np.array(image) / 255.0
            random_str = random_str + ' ' * (N_LEN - len(random_str))
            y[i] = [CHARACTERS.find(x) for x in random_str]
        return [X, y, input_length, label_length], np.ones(self.batch_size)
