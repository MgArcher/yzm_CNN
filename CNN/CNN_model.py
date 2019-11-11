# encoding: utf-8
"""
!/usr/bin/python3
@File: CNN.py
@Author:jiajia
@time: 2019/9/3 14:48
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

from keras.optimizers import Adam

from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint

from CNN.settings import *


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
        x = MaxPooling2D(2)(x)

    x = Flatten()(x)
    x = [Dense(n_class, activation='softmax', name='c%d' % (i + 1))(x) for i in range(n_len)]
    model = Model(inputs=input_tensor, outputs=x)
    return model


def watch_model(model, model_name=MODEL_NAME):
    model_name = model_name.replace('.h5', '.png')
    """模型可视化"""
    baseDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model')
    model_name = os.path.join(baseDir, model_name)
    plot_model(model, to_file=model_name, show_shapes=True)


def train_model(train_data, valid_data, model=None, model_name=MODEL_NAME):
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
    model_name = os.path.join(baseDir, model_name)
    cvs_name = model_name.replace('.h5', '.csv')
    if not model:
        model = build_model()

    # 设置训练的回调函数 1。超过三个epoch没有下降结束 2.记录每个loss的数值写入cnn.cvs,3.保存最优的模型
    callbacks = [
        EarlyStopping(patience=3),
        CSVLogger(cvs_name, append=True),
        ModelCheckpoint(model_name, save_best_only=True)
    ]
    # 编译模型  loss选择categorical_crossentropy（交叉熵损失函数)  Adam 优化器 * 评价函数metrics
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-3, amsgrad=True),
                  metrics=['accuracy'])

    if MULTITHERADING:
        # 使用多进程训练
        model.fit_generator(train_data, epochs=EPOCHS, validation_data=valid_data, workers=WORKERS,
                            use_multiprocessing=True, callbacks=callbacks)
    else:
        # 训练模型
        model.fit_generator(train_data, epochs=EPOCHS, validation_data=valid_data, callbacks=callbacks)
