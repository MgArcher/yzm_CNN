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
import keras.backend as K

from CTC.CTC_data import CaptchaSequence
from CTC.settings import *

mpl.rcParams['font.sans-serif'] = ['SimHei']   # 雅黑字体
# 加载模型
baseDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
model_path = os.path.join(baseDir, MODEL_NAME)
model = load_model(model_path, compile=False)

def once_file(file_path):
    """识别一张图片"""
    # 定义输入输出格式
    X = np.zeros((1, HEIGHT, WIDTH, 3), dtype=np.float32)
    # 打开图片转换成输入
    out = Image.open(file_path)
    # 改变大小 并保证其不失真
    out = out.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
    out = out.convert('RGB')
    X[0] = np.array(out) / 255.0
    # 识别内容
    y_pred = model.predict(X)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([CHARACTERS[x] for x in out[0]])

    plt.title('识别的内容为: %s' % out)
    plt.imshow(X[0], cmap='gray')
    plt.axis('off')
    plt.show()
    return out


def path_file(file_path):
    """识别多张图片"""
    # 定义输入输出格式
    files = os.listdir(file_path)
    X = np.zeros((len(files), HEIGHT, WIDTH, 3), dtype=np.float32)
    for i, file in enumerate(files):
        file_ = os.path.join(file_path, file)
        # 打开图片转换成输入
        out = Image.open(file_)
        # 改变大小 并保证其不失真
        out = out.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
        out = out.convert('RGB')
        X[i] = np.array(out) / 255.0
    # 识别内容
    y_pred = model.predict(X)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    Y = []
    for i, file in enumerate(files):
        Y.append(''.join([CHARACTERS[x] for x in out[i]]))
    return [{files[i]:Y[i]} for i in range(len(files))]


def create_once():
    data = CaptchaSequence(batch_size=1, steps=1)

    [X_test, y_test, _, _], _ = data[0]
    y_pred = model.predict(X_test)
    out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
    out = ''.join([CHARACTERS[x] for x in out[0]])
    y_test = ''.join([CHARACTERS[x] for x in y_test[0] if CHARACTERS[x] != ' '])

    plt.title('生成的验证码为：%s\n识别出的验证码为：%s' % (y_test, out))
    plt.imshow(X_test[0], cmap='gray')
    plt.axis('off')
    plt.show()

    return '生成的验证码为：%s\n识别出的验证码为：%s' % (y_test, out)


def create_accuracy(batch_num=100):
    data = CaptchaSequence(batch_size=1, steps=batch_num)
    batch_acc = 0
    for i in tqdm(range(len(data))):
        [X_test, y_test, _, _], _ = data[i]
        y_pred = model.predict(X_test)
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], )[0][0])[:, :4]
        out = ''.join([CHARACTERS[x] for x in out[0]])
        y_test = ''.join([CHARACTERS[x] for x in y_test[0] if CHARACTERS[x] != ' '])
        if out == y_test:
            batch_acc += 1
    return batch_acc / batch_num
