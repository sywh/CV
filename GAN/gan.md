# GAN
## CGAN
### base
- 作用
  - 一定程度上解决GAN生成结果的不确定性
- 例子：mnist
  - 原始GAN生成的图像是完全不确定的，具体生成的数字是几，完全不可控
  - 简单方法：为了让生成的数字可控，可以把数据集做一个切分，把数组0-9的数据集分别拆开训练训练9个模型，但这不仅分类麻烦，更重要的是，每一个类别的样本少，拿去训练GAN很可能导致欠拟合
  - CGAN：
    - Generator：输入不仅是随机噪声的采样z，还有欲生成图像的标签信息y，在mnist数据生成中，y就是一个one-hot向量
    - Discriminator：判别器的输入也包括样本的标签
### loss
- 公式：
- 理解：
  - loss设计和原始GAN基本一致
  - 只不过生成器和判别器的输入数据是一个条件分布
  - 具体编程实现时，只需要对随机噪声采样z和输入条件y做一个级联即可
## DCGAN
### base
- 问题
  - 对于视觉问题，如果使用基于DNN的GAN，则整个模型参数会非常巨大
  - 学习难读很大（低维度映射到高维度需要添加许多信息）
- 解决方案
  - DCGAN：将传统GAN的生成器和判别器均采用CNN实现
  - 使用了很多tricks
### tricks
- 使用CNN代替MLP
- 使用convolution代替pooling。其中，在discriminator上用strided convolution替代，在generator上用fractional-strided convolution替代
- 在generator和discriminator上都使用batchnorm
- 用global pooling代替全连接层，增加了模型稳定性，但损害了收敛速度
- generator输出层激活函数采用tanh，其它层使用relu
- disciminator所有层使用LeakeyReLU
### exp
- 了解输入随机噪声每一个维度代表的含义
  - 在隐空间上，假设知道哪几个变量控制着某个物体，那么将这几个变量挡住是不是就可以将生成图片中的某个物体消失？
  - 实验
    - 首先，生成150张图片，包括有窗户的和没窗户的
    - 然后，使用一个逻辑斯蒂回归函数来进行分类
    - 对于权重不为0的特征，认为它和窗户有关，将其挡住（如何将其挡住?），得到新的生成图片
  - 算数运算
    - 将几个输入噪声进行算术预案算，可以得到语义上进行算术运算的非常有趣的结果
    - 类似于word2vec