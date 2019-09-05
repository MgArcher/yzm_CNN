# encoding: utf-8
"""
!/usr/bin/python3
@File: test_model.py
@Author:jiajia
@time: 2019/9/3 18:00
"""
import os
from pylab import mpl

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import load_model

from make_data.CNN_data import CaptchaSequence
from settings import *

mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体


def once_file(file_path, model_name=MODEL_NAME, width=WIDTH, height=HEIGHT, characters=CHARACTERS):
    """识别一张图片"""
    # 加载模型
    baseDir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(baseDir.replace('model_test', 'model'), model_name)
    model = load_model(model_path)

    # 定义输入输出格式
    decode_y = lambda y: ''.join([characters[x] for x in np.argmax(np.array(y), axis=2)[:, 0]])
    X = np.zeros((1, height, width, 3), dtype=np.float32)

    # 打开图片转换成输入
    out = Image.open(file_path)
    # 改变大小 并保证其不失真
    out = out.resize((width, height), Image.ANTIALIAS)
    out = out.convert('RGB')
    X[0] = np.array(out) / 255.0
    # 识别内容
    y_pred = model.predict(X)
    Y = decode_y(y_pred)

    plt.title('识别的内容为: %s' % Y)
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')
    plt.show()

    return Y


def path_file(file_path, model_name=MODEL_NAME, width=WIDTH, height=HEIGHT, characters=CHARACTERS):
    """识别多张图片"""
    # 加载模型
    baseDir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(baseDir.replace('model_test', 'model'), model_name)
    model = load_model(model_path)

    # 定义输入输出格式

    # decode_y = lambda y: ''.join([characters[x] for x in np.argmax(np.array(y), axis=2)[:, 0]])
    #     # X = np.zeros((1, height, width, 3), dtype=np.float32)

    files = os.listdir(file_path)
    X = np.zeros((len(files), height, width, 3), dtype=np.float32)
    for i, file in enumerate(files):
        file_ = os.path.join(file_path, file)
        # 打开图片转换成输入
        out = Image.open(file_)
        # 改变大小 并保证其不失真
        out = out.resize((width, height), Image.ANTIALIAS)
        out = out.convert('RGB')
        X[i] = np.array(out) / 255.0
    # 识别内容
    y_pred = model.predict(X)
    Y = [''.join([characters[x] for x in np.argmax(np.array(y_pred), axis=2)[:, i]]) for i in range(len(files))]
    return [{files[i]:Y[i]} for i in range(len(files))]


def create_once(model_name=MODEL_NAME, characters=CHARACTERS):
    data = CaptchaSequence(batch_size=1, steps=1)
    decode_y = lambda y: ''.join([characters[x] for x in np.argmax(np.array(y), axis=2)[:, 0]])

    # 加载模型
    baseDir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(baseDir.replace('model_test', 'model'), model_name)
    model = load_model(model_path)

    X, y = data[0]
    y_pred = model.predict(X)

    plt.title('生成的验证码为：%s\n识别出的验证码为：%s' % (decode_y(y), decode_y(y_pred)))
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')
    plt.show()

    return '生成的验证码为：%s,识别出的验证码为：%s' % (decode_y(y), decode_y(y_pred))


def create_accuracy(model_name=MODEL_NAME, characters=CHARACTERS, batch_num=100):
    # 加载模型
    baseDir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(baseDir.replace('model_test', 'model'), model_name)
    model = load_model(model_path)
    batch_acc = 0

    with tqdm(CaptchaSequence(characters, batch_size=1, steps=batch_num)) as pbar:
        for X, y in pbar:
            y_pred = model.predict(X)
            y_pred = np.argmax(y_pred, axis=-1).T
            y_true = np.argmax(y, axis=-1).T
            batch_acc += (y_true == y_pred).all(axis=-1).mean()
    return batch_acc / batch_num
