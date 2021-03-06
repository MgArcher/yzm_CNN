# encoding: utf-8
"""
!/usr/bin/python3
@File: make_data.py
@Author:jiajia
@time: 2019/9/4 11:29
"""
import os
import random
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
from keras.utils import Sequence

from CNN.settings import *
from CNN.image_data import get_image


class CaptchaSequence(Sequence):
    def __init__(self, characters=CHARACTERS, batch_size=BATCH_SIZE, steps=STEPS, n_len=N_LEN, width=WIDTH, height=HEIGHT):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        # 定义X形状，Y形状
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = [np.zeros((self.batch_size, self.n_class), dtype=np.uint8) for _ in range(self.n_len)]
        img_generator = get_image()
        for i in range(self.batch_size):
            image, random_str = img_generator.__next__()
            X[i] = np.array(image) / 255.0
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, self.characters.find(ch)] = 1
        return X, y

