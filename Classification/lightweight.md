# lightweight
## introduction
- 由于神经网络的性质，为了获得更好的性能，网络层数不断增加，从7层的AlexNet到16层VGG，到22层GoogLeNet，再到152层ResNet，更有上千层的ResNet和DenseNet。虽然网络性能得到了提高，但随之而来的就是效率问题
- 效率问题主要是模型的**存储问题**和模型进行预测的**速度问题**
  - 存储问题：数百层网络有着大量的权值参数，保存大量权值参数对设备的内存要求很高
  - 速度问题：在实际应用中，往往是毫秒级别。为了达到实际应用标准，要么提高处理器性能，要么就减少计算量。而提高处理器性能在短时间内是无法完成的，因此减少计算量成为了主要的技术手段
- 对于效率问题，目前主要有两种常用的解决手段：**模型结构设计**和**模型压缩**
  - 模型压缩(Model Compression)：即在已经训练好的模型上进行压缩，使得网络携带更少的网络参数，从而解决内存问题，同时可以解决速度问题
  - 模型结构设计：主要思想在于设计更高效的网络计算方式（主要针对卷积方式），从而在网络参数减少的同时，不损失网络性能
## SqueezeNet
- 由伯克利&斯坦福发表于ICLR-2017
## MobileNetv1
- 由Google发表于CVPR-2017
- 创新点
  - 采用depthwise separable convolution代替传统卷积，以达到减少网络权值参数的目的
  - 提出两个超参数width multiplier和resolution multiplier来平衡时间和精度
- depthwise separable convolution
  - depthwise convolution：逐通道的卷积，一个卷积核负责一个通道，一个通道只被一个卷积核滤波
  - pointwise convolution：将depthwise convoluiton得到的feature map再串起来，前者只对输入进行了滤波，并没有把他们组合成新的特征。注意这个串是很重要的，**输出的每一个feature map要包含输入层所有feature map的信息**
- 计算量对比
  - 假设输入尺寸为DF*DF*M，标准卷积核的尺寸为DK*DK*M*N 
  - 标准卷积：步长为1，且padding，则输出尺寸为DF*DF*N，计算量为DK*DK*M*N*DF*DF
  - 深度可分离卷积
    - 先使用M个depthwise convolution filter对输入的M个通道分别进行卷积，得到尺寸为DF*DF*M，计算量为DK*DK*M*DF*DF
    - 再使用N个1×1×M的卷积核进行逐点卷积，得到输出尺寸为DF*DF*N，计算量为M*N*DF*DF
    - 总计算量为DK*DK*M*DF*DF + M*N*DF*DF
  - 计算量相比
    - 公式
    - 一般情况下N取值比较大，那么采用3×3卷积核的话，深度可分离卷积相对标准卷积可以降低大约9倍的计算量
- 网络结构对比
  - 标准卷积：3×3 Conv -> BN -> ReLU
  - 深度可分离卷积：3×3 Depthwise Conv -> BN -> ReLU -> 1×1 Conv -> BN -> ReLU
- 总结
  - 核心思想：采用depthwise convolution代替标准卷积，在相同的权值参数数量的情况下，可以减少数倍的计算量，从而达到提升网络运算速度的目的
  - depthwise convolution的思想非首创，借鉴于2014年一篇博士论文：L. Sifre. Rigid-motion scattering for image classification. hD thesis, Ph. D. thesis, 2014
  - 采用depthwise convolution会有一个问题，就是导致**信息流通不畅**，即输出的feature map仅包含输入的feature map的一部分，MobileNet采用了pointwise convolution解决这个问题
## MobileNetv2
- 在MobileNetv1基础上，结合ResNet结构思想，提出了MobileNetv3
- 创新点
  - Inverted residual block：引入残差结构和bottleneck层
  - Linear Bottlenecks：RELU会破坏信息，故去掉第二个Conv 1×1后的ReLU，改为线性神经元
- 标准残差和倒残差的区别
  - 标准残差：两头宽中间窄，倒残差两头窄，中间宽
  - 倒残差
    - 扩张层：1×1 Conv->BN->ReLU6，输入h*w*k，输出h*w*(tk)，t为扩张因子
    - 卷积层：3×3 depthwise with stride->BN->ReLU6，输入h*w*(tk)，输出h/s*w/s*(tk), s为步长
    - 压缩层：1×1 Conv->BN，输入h/s*w/s*(tk)，输出h/s*w/s*k'，输出和扩张层的输入相加
    - 针对stride=1和stride=2，block结构略有不同，在stride=2时，不采用shortcut
- MobileNetv2的网络结构
- 问题
  - 为什么倒残差要做成两头窄、中间宽的形式？
  - 为什么第二个1×1 Conv后面的ReLU要去掉？
  - bottleneck输入和输出通道数不同是怎么做到的？
  - t,c,n,s的设计有什么原则吗？
## MobileNetv3
- 在MobileNetv1和v2基础上，结合NAS搜索，提出了MobileNetv3
- 创新点
  - 网格搜索
    - 用NAS通过优化每个网络块来搜索全局网络结构
    - 用NetAdapt算法搜索每个层的过滤器数量
  - 人工设计
    - 更改末端计算量大的层，将增维的1×1层移到平均池化之后
    - 更改初始端为16个卷积核
    - 引入H-Swish激活函数
    - 加入SE模块
- H-Swish激活函数
  - 引入新的非线性激活函数h-swish，它是最近的swish非线性的改进版本，计算速度比swish快（比ReLU慢），更易于量化，精度上没有差异。网络越深越能够有效减少网络参数量（为什么？）。
  - swish：
    - 相比ReLU提高了精度，但因sigmoid函数而计算量大
    - 公式：swish(x) = x * sigmoid(x)
  - h-swish
    - 将sigmoid函数替换为分段线性硬模拟，使用的ReLU6在众多软硬件框架中都可以实现，量化时又避免了数值精度的损失，所以swish的硬版本变成了
    - 公式：h-swish(x) = x * ReLU6(x+3) / 6
- SE模块
  - V3基于V2的Inverted Residual and Linear Bottleneck，加入了Squeeze-and-Excite模块，压缩系数为4，精度最高
  - **SE让网络自动学习到了每个特征通道的重要程度**
  - SE瓶颈的大小与卷积瓶颈的大小有关，我们将他们全部替换为膨胀层通道数的1/4。这样做可以在适当增加参数数量的情况下提高精度，并且没有明显的延时成本
- 网络结构
  - MobileNetV3-Large：针对高资源情况下使用
  - MobileNetV3-Small：针对低资源情况下使用
- 总结
  - MobileNetV1：采用深度可分离卷积，大大提高了计算效率
  - MobileNetV2：引入了一个具有反向残差和线性瓶颈的资源高效模块，保护了低维空间的特征（？）
  - MobileNetV3：参考了三种模型：MobileNetV1的深度可分离卷积，MobileNetV2的具有线性瓶颈的反向残差结构(the inverted residual with linear bottleneck)，MnasNet+SE的自动搜索模型
## ShuffleNetv1
- 创新点
  - 分组逐点卷积(pointwise group conv)
  - 通道重排(channel shuffle)
- Channel Shuffle
  - 通道洗牌是介于整个通道的pointwise卷积和组内pointwise卷积的一种折中方案，传统策略是在整个feature map上执行1×1卷积。
  - 假设一个传统的深度可分离卷积由一个3×3的Depthwise卷积和一个1×1的Pointwise卷积组成，其中输入feature map的尺寸为h*w*c1，输出feature map的尺寸为h*w*c2，则Flops=9*h*w*c1 + h*w*c1*c2. 一般情况下c2是远大于9的，也就是说深度可分离卷积的性能瓶颈主要在Pointwise卷积上
  - 为了解决这个问题，ShuffleNetv1中提出仅在分组内进行Pointwise卷积，对于一个分成了g个组的分组卷积，其Flops=9*h*w*c1 + h*w*c1*c2/g。以上可以看出，组内Pointwise卷积可以非常有效地缓解性能瓶颈问题，然而这个策略的一个非常严重的问题是卷积之间的信息沟通不畅，网络趋近于一个由多个结构类似的网络构成的模型集成，精度大打折扣
  - 为了解决通道之间的沟通问题，ShuffleNetv1提出了其最核心的操作：通道洗牌(Channel Shuffle)。假设分组feature map的尺寸为w*h*c1，令c1=g*n，其中g表示分组的组数。Channel Shuffle的操作细节如下：
    - 将feature map展开成g*n*w*h的四维矩阵，为了简单理解，我们把w*h降到一维，表示为s
    - 沿着尺寸为g*n*s的矩阵的g轴和n轴进行转置（为什么要转置？）
    - 将g轴和n轴进行平铺后得到洗牌之后的feature map
    - 进行组内1×1卷积
- ShuffleNetv1单元
  - 带残差的深度可分离卷积：1×1 Conv->BN->ReLU->3×3 DWConv->BN->ReLU->1×1 Conv->BN
  - 上下两个部分的1×1卷积替换为1×1的分组卷积。分组g一般不会很大，论文中的值分别为1，2，3，4，8。g的值确保能被通道数整除，保证reshape操作的有效执行
  - 在第一个1×1卷积之后，添加一个Channel Shuffle操作
  - 对需要降采样的情况，左侧shortcut部分使用步长为2的3×3平均池化，右侧使用步长为2的3×3的depthwise卷积
  - 去掉了3×3 Depthwise卷积之后的ReLU激活，目的是为了减少ReLU激活造成的信息损耗
  - 如果进行了降采样，为了保证参数数量不骤减，往往需要加倍通道数量。所以降采样时shortcut使用拼接操作用于加倍通道数，不降采样时shortcut使用单位加
- 参考
  - 知乎：ShuffNet v1 和 ShuffleNet v2 https://zhuanlan.zhihu.com/p/51566209
- Flops计算
## ShuffleNetv2
- 背景
  - 目前大部分的模型加速和压缩文章在对比加速效果时用的指标都是FLOPs，这个指标主要衡量的就是卷积层的乘法操作。但是，本文通过一系列的实验发现FLOPs并不能完全衡量模型速度，相同FLOPs的网络实际速度差别可能很大，因此以FLOPs作为衡量模型速度的指标是有问题的。
  - 内存访问成本(memory access cost, MAC)对模型速度影响比较大，但难以在FLOPs指标中体现出来，对于GPUs来说可能会是瓶颈
  - 模型的并行程度也影响速度，并行度高的模型速度相对更快
  - 模型在不同平台上的运行速度是有差异的，如GPU和ARM，而且采用不同的库也会有影响
- 4条实用的指导原则
  - G1：使用相同的通道宽度的卷积，来最小化内存访问量MAC
    - 对深度可分离卷积中复杂度较高的1×1卷积，假设输入和输出特征通道数分别为c1和c2，特征图空间大小为h*w，那么FLOPs为B=hwc1c2，MAC为hw(c1+c2) + c1c2(这里假设内存足够，hwc1 是input内存大小，hwc2是output大小，c1c2是weight大小)
    - 根据均值不等式，固定B时，MAC存在下限，当c1=c2时，MAC取最小值
    - 实验证实，但通道比为1：1时，速度更快
  - G2：过量使用组卷积，会增加MAC
    - 组卷积(group convolution)可以减少复杂度却不损失模型容量，但分组过多会增加MAC。对于组卷积，FLOPs为B=hwc1c2/g(g是组数)，对应的MAC为hw(c1+c2) + c1c2/g
    - 固定输入hwc2以及B，发现当g增加时，MAC会同时增加
    - 实验证实，故不要使用太大g的组卷积
  - G3：网络碎片化(如Inception中的多路径)，会降低并行度
    - 一些网络如Inception，以及AutoML产生的网络NASNET-A，倾向于采用"多路"结构，即存在一个block中有很多不同的小卷积或者pooling，这很容易造成网络碎片化，降低模型的并行度，相应速度会变慢，这也可以通过实验得到证明
  - G4：减少元素级操作(如element-wise add)
    - 对于元素级操作，比如ReLU和Add，虽然它们的FLOPs较小，但却需要较大的MAC
    - 实验发现如果将ResNet中残差单元中的ReLU和shortcut移除的话，速度有20%的提升
- ShuffleNetv1的缺陷
  - v1大量使用了1×1组卷积，违背了G2原则
  - v1采用了类似ResNet的瓶颈层，输入和输出通道数不同，违背了G1原则
  - 使用过多的组，违背了G3原则（为什么？）
  - 短路连接中存在大量的元素级add运算，违背了G4原则
- ShuffleNetv2的改进
  - 引入channel split，在开始时先将输入特征图在通道维度分成两个分支，通道数分别为c'和c-c'，实际实现时c'=c/2。左边分支做同等映射，右边分支包含3个连续卷积，并且输入和输出的通道相同，这符合G1；而且两个1×1卷积不再是组卷积，这符合G2；两个分支的输出不再是Add元素，而是concat在一期，紧接着是对两个分支concat结果进行channel shuffle，以保证两个分支信息交流。其实concat和channel shuffle可以和下一个模块单元的channel split合成一个元素级运算，这符合G4
  - 对于下采样模块，不再有channel split，而是每个分支都是直接copy一份输入，每个分支都有stride=2的下采样，最后concat在一起后，特征图空间大小减半，但是通道数翻倍
  - 值得注意的是，v2在全局pooling之前增加了conv5卷积，这是与v1的一个区别。
  - shuffleNetv2结构基本与v1类似，通过设定每个block的channel数，如0.5x，1x，可以调整模型的复杂度
- 借鉴
  - ShuffleNetv2借鉴了DenseNet网络，把shortcut结构从Add换成了Concat，这实现了**特征重用**
  - 不同于DenseNet，v2并不是密集地concat，而是concat之后有channel shuffle以混合特征，这或许是v2即快又好的一个重要原因 
## Xception
## reference
- 知乎：常用的轻量级神经网络结构总结 https://zhuanlan.zhihu.com/p/95576037
- 知乎：ShuffleNetV2：轻量级CNN网络中的桂冠 https://zhuanlan.zhihu.com/p/48261931
- https://cloud.tencent.com/developer/article/1451558
- https://www.sohu.com/a/215356462_465975