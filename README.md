本文完全参考自：https://github.com/ypwhs/captcha_break
需要安装的模块在requirements.txt中
使用方式
运行train.py 训练模型
CaptchaSequence是使用python自动生成验证码
FileCaptchaSequence是使用本地文件进行训练
需要修改的参数都在settings.py文件中

运行test.py测试模型
提供一下方式进行测试
读取文件方式测试
读取文件夹方式测试
生成验证码方式测试
测试准确率（生成验证码方式）



实现流程如下图所示：

此图来源于：https://blog.csdn.net/qq_32791307
这个图已经把流程说的明明白白了，整个流程就如上图所示
构造模型
首先我们先要构造模型：
根据我们的需求是想识别验证码，根据这个需求可知咱们想要的模型输入是一个图片，而输出是字符串。
那么我们在构造模型的时候就需要指定输入和输出的大小。对于模型来说输入输出的都是矩阵，如何把图像转换为矩阵之后再说，首先我们需要定义输入，再代码中形式为：
其中height就是图形的高，width为图像的宽，3是因为图像都是三种颜色构成的，所以指定其为三维。
input_tensor = Input((height, width, 3))

定义好输入层后就要开始要把图像进行卷积了，这样才能更好的寻找其对应的特征并且训练起来更快。那么我们就要开始定义卷积层了。针对于图像的卷积，kears提供了一个函数
Conv2D
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)

2D卷积层（例如，图像上的空间卷积）。
该层创建一个卷积内核，该卷内核与层输入卷积以产生一个输出张量。如果use_bias为True，则创建偏置向量并将其添加到输出。最后，如果 activation不是None，它也会应用于输出。
将此图层用作模型中的第一个图层时，请提供关键字参数input_shape （整数元组，不包括批处理轴），例如，input_shape=(128, 128, 3)对于128x128 RGB图片data_format="channels_last"。
其中用到的参数
filters：整数，输出空间的维数（即卷积中输出滤波器的数量）（卷积核数目）。
kernel_size：2个整数的整数或元组/列表，指定2D卷积窗口的高度和宽度。可以是单个整数，以指定所有空间维度的相同值。
padding：一个"valid"或"same"（不区分大小写）。请注意，"same"是跨越与后端略微不一致strides！= 1，
kernel_initializer：kernel权重矩阵的初始化器（参见初始化器）。
卷积后对其进行归一化，并指定relu为其激活函数



定义卷积核数目，每个卷积核卷积两次然后进行池化。（卷积到某个值为一为止，最好根据图像大小修改卷积层数）




模型结构很简单，特征提取部分使用的是两个卷积，一个池化的结构，这个结构是学的 VGG16 的结构。

卷积完之后就压扁输入：
x = Flatten()(x)

展平输入。不影响批量大小。

卷积完了之后创建全连接层：

用来输出识别到每个字符即输出有四个：


模型结构：


数据处理
模型确定完毕了，那么就要根据这个模型定义的入口大小，把我们的图像改成模型能接受的数字矩阵，这里实现两种方式给数据，一种就通过python自动生成的方式生成验证码，另一种是读取本地存储的图像。
首先我们要定义输入输出的形状：

X 定义了输入的形状
self.batch_size 定义了输入参数中是几张图片构成的矩阵
self.height ，self.width 3,定义了RGB图像的大小结构
合起来构成输入参数的矩阵
y定义了输出的形状

y是个列表，对应着每个分类器的输出
self.batch_size 定义了输入有多少长图片
self.n_class 定义了有多少种类别，在本次实验中输出为0-9+a-z共36种输出，按照顺序定义其位置处为1，其余的为0。

输入输出的结构确定下来了那么我们就要准备一个数据生成器，为了训练做准备了，训练数据有两种办法，一个是自己生成验证码，一个是使用标注好的验证码，标注方式为名称为验证码内容_标识码的方式。然后，我们就需要把图片转换为模型需要的参数了。
把相应的图片文件和内容转换成矩阵并封装成生成器供模型使用就好了。
代码在 make_data文件夹中。

训练模型
使用keras训练模型的步骤就非常容易了，第一步设置几个训练需要用到的回调函数


然后编译模型：

最后给出训练集和验证集，就开始训练模型了：


验证模型
训练完毕之后，需要测试一下模型的效果，经过使用100张验证码图片验证后发现其准确率达到了96%，证明此模型是有用的，但是绘制loss图表发现其loss下降很慢，还有优化的空间。

