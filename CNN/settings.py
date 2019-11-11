# encoding: utf-8
"""
!/usr/bin/python3
@File: settings.py
@Author:jiajia
@time: 2019/9/4 10:14
"""
# 控制使用多线程模式
MULTITHERADING = False
# 设置线程数
WORKERS = 2
# 训练迭代次数
EPOCHS = 100
# 模型保存名称
MODEL_NAME = 'cnn_best.h5'
############
# 生成验证码的参数#
##############
# 为True时使用文件夹中数据进行训练，为False为生成方式进行训练
FILE = False
# 每一批次使用的验证码数量
BATCH_SIZE = 64
# 一次迭代使用验证码的批次(只包含生成的验证码，本地验证码为选定文件夹下的所有文件)
STEPS = 1000

# 验证码文件所在路径（需要标注好的，标注方式为“验证码内容_序列号.jpg”）
PATH = 'pic'

# 要识别的字符
CHARACTERS = "0123456789" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# 图片宽度
WIDTH = 200
# 图片高度
HEIGHT = 60
# 一张图片上字符数
N_LEN = 4
# 字符种类数目
N_CLASS =len(CHARACTERS)