# encoding: utf-8
"""
!/usr/bin/python3
@File: CNN_test.py
@Author:jiajia
@time: 2019/11/7 10:36
"""
# encoding: utf-8
"""
!/usr/bin/python3
@File: test.py
@Author:jiajia
@time: 2019/9/3 17:21
"""
from CNN.test_model import *


if __name__ == '__main__':
    # 测试模型
    # 读取文件方式测试
    test = once_file(r'pic\0CR3_00638.jpg')
    print(test)
    # 读取文件夹方式测试
    test = path_file(r'pic')
    print(test)
    # 生成验证码方式测试
    test = create_once()
    print(test)
    # 测试准确率
    accuracy = create_accuracy()
    print(accuracy)
