# encoding: utf-8
"""
!/usr/bin/python3
@File: cs.py
@Author:jiajia
@time: 2019/11/4 10:16
"""
import matplotlib.pyplot as plt
import keras.backend as K
import numpy as np
from tqdm import tqdm
from captcha.image import ImageCaptcha
import random
import string
from keras.models import *
from keras.layers import *
from keras.utils import Sequence

from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.optimizers import *
from keras.callbacks import Callback
# 初始参数
characters = string.digits + string.ascii_uppercase + " "
width, height, n_len, n_class = 128, 64, 4, len(characters) + 1
print(characters)
#验证
# 创建训练集
class CaptchaSequence(Sequence):
    def __init__(self, characters, batch_size, steps, n_len=4, width=128, height=64,
                 input_length=16, label_length=4):
        self.characters = characters
        self.batch_size = batch_size
        self.steps = steps
        self.n_len = n_len
        self.width = width
        self.height = height
        self.input_length = input_length
        self.label_length = label_length
        self.n_class = len(characters)
        self.generator = ImageCaptcha(width=width, height=height)

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        X = np.zeros((self.batch_size, self.height, self.width, 3), dtype=np.float32)
        y = np.zeros((self.batch_size, self.n_len), dtype=np.uint8)
        input_length = np.ones(self.batch_size) * self.input_length
        label_length = np.ones(self.batch_size) * self.label_length
        for i in range(self.batch_size):
            # random_str = ''.join([random.choice(self.characters) for j in range(self.n_len)])
            random_str = ''.join([random.choice(self.characters) for j in range(random.randint(2, self.n_len))])
            random_str = '2A'
            random_str = random_str + ' ' * (self.n_len - len(random_str))
            X[i] = np.array(self.generator.generate_image(random_str)) / 255.0
            y[i] = [self.characters.find(x) for x in random_str]
        return [X, y, input_length, label_length], np.ones(self.batch_size)


train_data = CaptchaSequence(characters, batch_size=4, steps=8000)
valid_data = CaptchaSequence(characters, batch_size=4, steps=8000)

base_model = load_model('model/ctc.h5')

characters2 = characters + ' '
[X_test, y_test, _, _], _  = train_data[0]
y_pred = base_model.predict(X_test)
out = K.get_value(K.ctc_decode(y_pred, input_length=np.ones(y_pred.shape[0])*y_pred.shape[1], )[0][0])[:, :4]
print(out)
out = ''.join([characters[x] for x in out[0]])

y_true = ''.join([characters[x] for x in y_test[0]])

plt.imshow(X_test[0])
plt.title('pred:' + str(out) + '\ntrue: ' + str(y_true))
plt.show()
# argmax = np.argmax(y_pred, axis=2)[0]
# list(zip(argmax, ''.join([characters2[x] for x in argmax])))