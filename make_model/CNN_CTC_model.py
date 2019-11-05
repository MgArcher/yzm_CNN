# encoding: utf-8
"""
!/usr/bin/python3
@File: CNN_CTC_model.py
@Author:jiajia
@time: 2019/9/6 15:30
"""
import os

from keras.utils import plot_model

from keras.models import Input
from keras.models import Model

from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

from keras.layers import Permute
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import Lambda

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

from settings import *


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def build_model(width=WIDTH, height=HEIGHT, n_len=N_LEN, n_class=N_CLASS):
    """
    构建深度卷积神经网络
    特征提取部分使用的是两个卷积，一个池化的结构，这个结构是学的 VGG16 的结构。
    width: 图片的宽度
    height:图片的宽度
    默认为RGB模式的图像,有三层

    n_len: 定义全连接层个数,即要识别的字符数
    n_class: 定义了字符的类别数,即识别的字符有多少种
    """
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i, n_cnn in enumerate([2, 2, 2, 2, 2]):
        for j in range(n_cnn):
            x = Conv2D(32 * 2 ** min(i, 3), kernel_size=3, padding='same', kernel_initializer='he_uniform')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
        x = MaxPooling2D(2 if i < 3 else (2, 1))(x)

    x = Permute((2, 1, 3))(x)
    x = TimeDistributed(Flatten())(x)

    rnn_size = 128
    x = Bidirectional(GRU(rnn_size, return_sequences=True))(x)
    x = Bidirectional(GRU(rnn_size, return_sequences=True))(x)
    x = Dense(n_class, activation='softmax')(x)

    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[n_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=loss_out)
    return base_model
