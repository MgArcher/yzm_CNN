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
# import string
# characters = string.digits + string.ascii_uppercase
# 要识别的字符
CHARACTERS = "0123456789" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ" + " "
# 图片宽度
WIDTH = 300
# 图片高度
HEIGHT = 80
# 一张图片上最多的字符数
N_LEN = 8
# 字符种类数目
N_CLASS =len(CHARACTERS)
# 输入模型的序列长度（大于最多字符数就行）
INPUT_LENGHT = 16



# 训练迭代次数
EPOCHS = 100

# 模型保存名称
MODEL_NAME = 'ctc_best.h5'

############
# 生成验证码的参数#
##############

# 每一矩阵使用的验证码数量
BATCH_SIZE = 4
# 一次生成器迭代使用验证码的数目(只包含生成的验证码，本地验证码为选定文件夹下的所有文件)
STEPS = 4000

# 验证码文件所在路径（需要标注好的，标注方式为“验证码内容_序列号.jpg”）
PATH = 'pic'