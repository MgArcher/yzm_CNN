# encoding: utf-8
"""
!/usr/bin/python3
@File: CTC_test.py
@Author:jiajia
@time: 2019/11/7 15:20
"""
from CTC.test_model import *


if __name__ == '__main__':
    # 测试模型
    # 读取文件方式测试
    # test = once_file(r'pic\0CR3_00638.jpg')
    # print(test)
    # 读取文件夹方式测试
    # test = path_file(r'pic')
    # print(test)
    # 生成验证码方式测试
    test = create_once()
    print(test)
    # 测试准确率
    # accuracy = create_accuracy()
    # print(accuracy)
