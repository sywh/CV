# Convolution (Neural Network)
## Base
- 定义
  - 卷积操作是对输入矩阵和卷积核进行点乘求和的数学操作，求得的结果为原始图像中提取的特定局部特征。
- 作用
  - 提取图像的特征，不同层次的卷积操作提取到的特征类型是不同的
    - 浅层卷积：提取边缘特征
    - 中层卷积：提取局部特征
    - 深层卷积：提取全局特征
  - 卷积核作用
    - 输出原图
    - 边缘检测（突出边缘差异）
    - 边缘检测（突出中间值）
    - 图像锐化
    - 方块模糊
    - 高斯模糊
- 基础
  - 卷积层参数
    - 卷积核大小(Kernel Size)：定义了卷积的感受野
    - 卷积核步长(Stride)
    - 填充方式(Padding)
    - 输入通道数(In Channels)：卷积核的深度
    - 输出通道数(Out Channels)：卷积核的个数
  - 卷积类型
    - 标准卷积：提取相邻像素之间的关联关系，3×3的卷积核可以获得3×3的感受野
    - 扩张卷积（空洞卷积）：引入扩张率(Dilation Rate)参数，使用同样尺寸的卷积核可以获得更大的感受野；相应地，在相同感受野的签一下比标准卷积采用更少的参数。3×3的卷积核可以获得5×5的感受野。在实时图像分割领域广泛应用
    - 转置卷积（反卷积）：反卷积是一种特殊的正向卷积，先按照一定的比例通过补0来扩大输入图像的尺寸，接着旋转卷积核，再进行正向卷积。输入到输出的维度变换关系恰好与普通卷积的变换关系相反。转置卷积常见于目标检测领域中对小目标的检测和分割领域还原输入图像尺度。
    - 可分离卷积：通常应用在模型压缩或轻量网络中。仔细看看细节？
  - 3D卷积：有空看一下？
  - 1×1卷积
    - 应用：
      - NIN中用1×1卷积代替MLP，实现信息的跨通道交互和整合
      - GoogLeNet中在较大卷积核的卷积层前引入1×1卷积，在不改变模型表达能力的前提下大大减少模型参数量
    - 作用
      - 实现信息的跨通道交互和整合
      - 对卷积核通道数进行降维和升维，减少参数量
## Discussion
### 卷积层和池化层有什么区别
- 结构：
  - 卷积层：零填充时输出维度不便，而通道数改变
  - 池化层：通常特征维度会降低，通道数不变
- 稳定性：
  - 卷积层：输入特征发生细微改变时，输出结果会改变
  - 池化层：感受野内的细微变化不影响输出结果
- 作用： 
  - 卷积层：感受野内提取局部关联特征
  - 池化层：感受野内提取泛化特征，降低维度
- 参数量：
  - 卷积层：与卷积核尺寸、卷积核个数相关
  - 池化层：不引入额外参数
### 卷积核是否越大越好
- 大卷积核
  - 历史：早期卷积神经网络(如LeNet-5, AlexNet)，用到了比较大的卷积核(11*11, 5*5)。受限于当时的计算能力和模型结构的设计，无法将网络叠加得很深，因此卷积层需要设置较大的卷积核以获取更大的感受野。但是，大卷积核会导致计算量大幅增加，不利于训练更深层的模型。
  - 适合范围：在NLP领域，文本不像图像可以对特征进行很深层的抽象，故CNN通常较浅，而文本特征有时又需要较广的感受野来让模型组合更多的特征(如词组和字符)，此时采用较大的卷积核是更好的选择
- 小卷积核
  - 历史：后来的卷积神经网络(如VGG，GoogLeNet)，发现堆叠2个3*3卷积核可以获得与5*5卷积核相同的感受野，同时参数量更少。因此，大多数情况下通过堆叠较小的卷积核比直接用单个更大的卷积核更有效
  - 适用范围：大多数情况。但卷积核太小不合适，如1×1卷积核不能对输入特征进行有效的组合。另一方面，卷积核太大则会组合过多的无意义特征从而浪费计算资源。
### 每层卷积是否只能用一种尺寸的卷积核
- GoogLeNet，Inception每层均使用了多个卷积核结构，再将分别得到的特征进行整合，得到的新特征可以看作不同感受野提取的特征组合，相比单一卷积核具有更强的表达能力。
### 如何减少卷积层参数量
- 堆叠小卷积核代替大卷积核
- 使用分离卷积操作
- 使用1×1卷积操作降低channel数
- 在卷积前使用池化操作
### 提高CNN泛化能力的方法
- 使用更多数据：更多的数据可以让模型得到充分的学习，也更容易提高泛化能力
- 使用更大批次：在相同迭代次数和学习率的条件下，每批次采用更多的数据有助于模型更好地学到正确的模式，模型输出结果也更稳定
- 调整数据分布：大多数场景下的数据分布是不均横的，模型过多地学习某类数据容易导致其输出结果偏向于该类型数据，此时通过调整输入的数据分布可以一定程度提高泛化能力
- 调整目标函数：某些情况下，目标函数的选择会影响模型的泛化能力
- 调整网络结构：浅层CNN参数量较少容易导致欠拟合；深层CNN若没有充足的训练数据则容易导致过拟合
- 数据增强：需要注意的是数据变化应尽可能不破坏元数据的主体特征（如在图像分类任务中，对图像进行裁剪时，不能将分类主体目标裁出边界）
- 权值正则化：在损失函数中添加一项权重矩阵的正则项作为惩罚项，用来惩罚权重过大的情况
- 屏蔽网络节点：可认为是网络结构上的正则化
### CNN的特点
- 区域不变性：滤波器在每层的输入图像上滑动，检测的是局部信息，然后通过pooling取最大值或均值。pooling这步综合了局部特征，失去了每个特征的位置信息。这很适合基于图像的任务，比如要判断一幅图里有没有猫这种生物，你可能不会关心这只猫出现在图像的哪个区域。但在NLP里，词语在句子或者段落里出现的位置、顺序，都是很重要的信息。
- 局部组合性：CNN中，每个滤波器都把较低层的局部特征组合成较高层的全局化的特征。这在CV里很好理解，像素组合成边缘，边缘生成形状，最后把各种形状组合起来得到复杂的物体表达。在语言里，当然也有类似的组合关系，但是远不如图像来的直接。而且，在图像里，相邻像素必须是相关的，而相邻的词语却未必相关。
- 局部连接：CNN在空间维度上是局部链接，在深度上是全链接。二维图像局部像素关联性较强，局部链接保证了训练后的滤波器能够对局部特征有最强的响应，使神经网络可以提取数据的局部特征，减少参数量。
- 权值共享：卷积核权重对于同一深度切片的神经元是共享的。权重共享在一定程度上是有意义的。在神经网络中，提取的底层边缘特征与其在图中的位置无关；但在另一些场景中是无意的，如在人脸识别任务，我们期望在不同的位置学到不同的特征。权重共享的好处是大大减少了网络的参数，降低了网络的训练难度。
- 特征提取器：CNN的强大之处在于它的多层结构能自动学习特征，并且可以学习到多个层次的特征：较浅的卷积层感知域较小，学习到一些局部区域的特征；较深的卷积层具有较大的感知域，能够学习到更加抽象的一些特征。高层的抽象特征对物体的大小、位置和方向等敏感性更低，从而有助于识别性能的提高。所以，我们常常可以将卷积层看作是特征提取器。
## Problem
- 反卷积操作的理解？参见知乎答案如何理解深度学习中的deconvolution networks
- 空洞卷积的理解？
- 权值正则化的理解？为什么惩罚权重过大的项？
- 为什么CNN在图像处理领域有广泛的应用？
## Remain
- NetVLAD池化
- 局部卷积在人脸中的应用
- 棋盘效应