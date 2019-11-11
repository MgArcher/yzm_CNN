# encoding: utf-8
"""
!/usr/bin/python3
@File: imafe_data.py
@Author:jiajia
@time: 2019/11/11 11:14
"""
import os
import random
from CNN.settings import *
from captcha.image import ImageCaptcha
from PIL import Image


def get_image():
    """
    :param fixed_length: 调解产生的验证码图片中字符数是定长还是不定长度的，为True是定长的，字符数为n_len的值，为False是不定长度的,字符数为2-n_len的区间
    :param width: 图片的宽度
    :param height: 图片的高度
    :param n_len: 图片字符数
    :return: 验证码图片，图片中的内容
    """
    # 一张图片上最多的字符数
    if FILE:
        baseDir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), PATH)
        files = os.scandir(baseDir)
        while True:
            try:
                file = files.__next__()
            except:
                files = os.scandir(baseDir)
                file = files.__next__()
            random_str = file.name.split('_')[0]
            # 打开图片转换成输入
            out = Image.open(file.path)
            # 改变大小 并保证其不失真
            out = out.resize((WIDTH, HEIGHT), Image.ANTIALIAS)
            out = out.convert('RGB')
            yield out, random_str
    else:
        img = ImageCaptcha(width=WIDTH, height=HEIGHT)
        characters = CHARACTERS.replace(' ', '')
        while True:
            random_str = ''.join([random.choice(characters) for _ in range(N_LEN)])
            image = img.generate_image(random_str)
            yield image, random_str


