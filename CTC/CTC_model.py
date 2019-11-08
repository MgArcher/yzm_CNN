# encoding: utf-8
"""
!/usr/bin/python3
@File: CNN_CTC_model.py
@Author:jiajia
@time: 2019/9/6 15:30
"""
import os
import numpy as np
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
import keras.backend as K
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from CTC.settings import *


# 构造评估函数
def evaluate(base_model, valid_data):
    batch_acc = 0
    for [X_test, y_test, _, _], _ in valid_data:
        y_pred = base_model.predict(X_test)
        shape = y_pred.shape
        out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(shape[0])*shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc / STEPS


class Evaluate(Callback):
    def __init__(self, base_model, valid_data):
        self.accs = []
        self.base_model = base_model
        self.valid_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        acc = evaluate(self.base_model, self.valid_data)
        logs['val_acc'] = acc
        self.accs.append(acc)
        print(f'\nacc: {acc * 100:.4f}')


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
    return model, base_model


def train_model(train_data, valid_data, model=None, epochs=EPOCHS, model_name_=MODEL_NAME, MULTITHERADING=MULTITHERADING, workers=WORKERS):
    """
    :param train_data:训练集
    :param valid_data: 验证集
    :param model: 模型
    :param epochs: 迭代代数
    :return:
    """
    """
    训练模型
    训练模型反而是所有步骤里面最简单的一个，直接使用 model.fit_generator 即可，这里的验证集使用了同样的生成器，由于数据是通过生成器随机生成的，所以我们不用考虑数据是否会重复。
    为了避免手动调参，我们使用了 Adam 优化器，它的学习率是自动设置的，我们只需要给一个较好的初始学习率即可。
    EarlyStopping 是一个 Keras 的 Callback，它可以在 loss 超过多少个 epoch 没有下降以后，就自动终止训练，避免浪费时间。
    ModelCheckpoint 是另一个好用的 Callback，它可以保存训练过程中最好的模型。
    CSVLogger 可以记录 loss 为 CSV 文件，这样我们就可以在训练完成以后绘制训练过程中的 loss 曲线。
    注意，这段代码在笔记本电脑上可能要较长时间，建议使用带有 NVIDIA 显卡的机器运行。注意我们这里使用了一个小技巧，添加 workers=4 参数让 Keras 自动实现多进程生成数据，摆脱 python 单线程效率低的缺点。
    """
    baseDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    model_name = os.path.join(baseDir, model_name_)
    cvs_name = model_name.replace('.h5', '.csv')
    loss_mode_name = os.path.join(baseDir, 'loss_' + model_name_)
    model, base_model = build_model()

    callbacks = [EarlyStopping(patience=3), CSVLogger(cvs_name), ModelCheckpoint(loss_mode_name, save_best_only=True)]

    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=Adam(1e-3, amsgrad=True))
    if MULTITHERADING:
        model.fit_generator(train_data, epochs=epochs, validation_data=valid_data, workers=workers,
                            use_multiprocessing=True, callbacks=callbacks)
    else:
        model.fit_generator(train_data, epochs=epochs, validation_data=valid_data, callbacks=callbacks)
    # 保存
    base_model.save(model_name, include_optimizer=False)
