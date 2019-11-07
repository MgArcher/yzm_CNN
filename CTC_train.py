# encoding: utf-8
"""
!/usr/bin/python3
@File: train.py
@Author:jiajia
@time: 2019/9/4 16:21
"""
from multiprocessing import freeze_support

from CTC.CTC_model import *
from CTC.CTC_data import *
from CTC.settings import STEPS


if __name__ == '__main__':
    # 确保线程安全(非多线程模式可注释掉)
    freeze_support()
    # 读取文件方式
    # # 训练集
    train_data = FileCaptchaSequence()
    # 测试集
    valid_data = FileCaptchaSequence()

    # 自动生成方式
    # 训练集
    # train_data = CaptchaSequence()
    # # 测试集 取训练集中1/10作为测试集
    # valid_data = CaptchaSequence(steps=int(STEPS / 10))

    # 训练模型
    train_model(train_data, valid_data)

