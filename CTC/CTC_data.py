# encoding: utf-8
"""
!/usr/bin/python3
@File: CTC_data.py
@Author:jiajia
@time: 2019/11/6 16:42
"""
import os
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
from keras.utils import Sequence

from CTC.settings import *


# 创建训练集
class CaptchaSequence(Sequence):
    def __init__(self, characters=CHARACTERS, batch_size=BATCH_SIZE, steps=STEPS, n_len=N_LEN, width=WIDTH, height=HEIGHT,
                 input_length=INPUT_LENGHT, label_length=N_LEN):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_len), dtype=np.uint8)
        input_length = np.ones(self.batch_size) * self.input_length
        label_length = np.ones(self.batch_size) * self.label_length
        for i in range(self.batch_size):
            # random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            random_str = ''.join([random.choice(self.characters[:-1]) for _ in range(random.randint(2, self.n_len))])
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            random_str = random_str + ' ' * (self.n_len - len(random_str))
            y[i] = [self.characters.find(x) for x in random_str]
        return [X, y, input_length, label_length], np.ones(self.batch_size)


class FileCaptchaSequence(Sequence):
    def __init__(self, characters=CHARACTERS, batch_size=BATCH_SIZE, steps=STEPS, n_len=N_LEN, width=WIDTH, height=HEIGHT, input_length=INPUT_LENGHT, label_length=N_LEN, path=PATH):
        self.path = path
        self.files = os.scandir(path)
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        # 定义X形状，Y形状
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_len), dtype=np.uint8)
        input_length = np.ones(self.batch_size) * self.input_length
        label_length = np.ones(self.batch_size) * self.label_length
        for i in range(self.batch_size):
            try:
                file = self.files.__next__()
            except:
                self.files = os.scandir(self.path)
                file = self.files.__next__()
            random_str = file.name.split('_')[0]
            random_str = random_str + ' ' * (self.n_len - len(random_str))
            # 打开图片转换成输入
            out = Image.open(file.path)
            # 改变大小 并保证其不失真
            out = out.resize((self.width, self.height), Image.ANTIALIAS)
            out = out.convert('RGB')
            X[i] = np.array(out) / 255.0
            y[i] = [self.characters.find(x) for x in random_str]
        return [X, y, input_length, label_length], np.ones(self.batch_size)
