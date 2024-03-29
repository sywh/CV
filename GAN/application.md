# application
## image translation
### base
- 引入cGAN中给的条件是相对弱的条件，比如标签，如果直接用图片作为条件行不行？比如以素描图为条件，生成对应的照片？这种问题被定义为图像翻译，图像翻译就是建立一类图到另一类图的映射关系。典型的图像翻译问题比如从素描图到照片的映射，从黑白图到彩色图的映射。
- 定义：图像翻译，指从一副图像到另一副图像的转换。可以类比机器翻译，从一种语言转换为另一种语言。
- 常见图像翻译任务
  - 图像去噪
  - 图像超分辨
  - 图像补全
  - 风格迁移
- 分类
  - 有监督图像翻译：原始域与目标域存在一一对应数据
  - 无监督图像翻译：原始域与目标域不存在一一对应数据
### pix2pix
- 设计
  - 直观想法：设计一个CNN网路，直接建立输入-输出的映射，就像图像去噪问题一样。
    - 问题：生成图像质量不清晰
    - 原因：比如分割图->街景图，语义分割图的每个标签比如"汽车"可能对应不同样式、颜色的汽车，那么模型学习到的会是所有不同汽车的平均，这样会造成模糊
  - pix2pix
    - 思路：在上述直观想法的基础上加入一个判别器，判断输入图片是否是真实样本。通过加入GAN的loss去惩罚模型, 解决生成图像的模糊问题
    - 基于CGAN：和CGAN有所不同，输入只有条件信息。原始的CGAN需要输入随机噪声，以及条件。之所以没有输入噪声信息，是因为在实际实验中，噪声往往被淹没在条件当中，所以这里直接省去了。
    - loss:
      - GAN loss：LSGAN的最小二乘loss，并使用PatchGAN来进一步保证生成图像的清晰度。PatchGAN将图像划分成很多个Patch，并对每一个Patch使用判别器进行判别（实际代码实现有更取巧的办法），将所有Patch的loss求平均作为最终的loss
      - 输出和标签的L1 loss：采用L1 loss而不是L2 loss的理由很简单：L1 loss相比L2 loss保边缘(L1 loss基于拉普拉斯先验，L2 loss基于高斯先验)
    - 测试时也使用Dropout，以使输出多样化
- 问题
  - 如何生成高分辨率图像和高分辨率视频
    - 在pix2pix这一通用的图像翻译框架上，利用更好的网络结构以及更多的先验
    - pix2pixHD：提出用多尺度的生成器和判别器，来生成高分辨率的图像
    - vid2vid：在pix2pixHD基础上，利用光流、时序约束等生成高分辨率视频。
  - 有监督图像翻译的缺点
    - 需要一一对应的图像
- 特点
  - 判别器输入：Pix2Pix判别器输入其实是比较特殊的，它的输入不是真实图片或生成图片，而是把素描图和真实图片或生成图片拼接以后，再送入判别器。作者实验中发现这么对效果有提升，原因是这么做其实**让判别器能学到更多的东西**：判别器不仅能用来区分真实图片和生成图片，还能区分真实图片和生成图片是否与素描图match。真实图片y和素描图x是match的，如果生成器生成的G(x)和x不match，那判别器一下就判断出来生成的图片是假的，就会产生梯度让生成器生成和素描图x match的图片
- 优缺点
  - 优点：Pix2Pix重要性在于，在这之前，不同的图像翻译任务需要用不同的模型去生成，在Pix2Pix出现后，这些任务可以用同一种框架来解决
  - 缺点：
    - 训练的时候需要paired data：如果没有paired data，会怎么样呢？我们先看看没有paired data，按上述框架，要做什么改动？因为没有paired data，所以y和x是不匹配的，就没必要把x也输入到判别器中。好像也能训，但效果会非常差。原因丢失了素描图x对生成图的约束关系。实际上，Pix2Pix除了一般的GAN loss，还有一个L1 loss，是计算G(x）和真实图片y之间的L1损失，这个L1损失同样会对生成图G（x)进行约束，使其和真实图片y尽可能match。所有没有paired data，G(x)即不能用素描图x进行约束，也不能用真实图片y进行约束，缺失了上面两个约束以后，训练时会容易导致一些严重的问题，就是模式崩溃。而且，生成的图和素描图x也没有强对应关系。
### CycleGAN
- 引入
  - 前面说Pix2Pix必须要paired data，这极大限制了Pix2Pix的应用，因为paired data获取成本是很高的，比如要获得白天和黑夜的paired data，就需要对同一个场景的白天和黑夜分别采集照片。而且有的时候根本就没有paired data，比如现实生活中没有马和斑马的paired data
  - 所以我们需要有一个不需要paired data的图像翻译框架
- 思路
  - CycleGAN为了解决paired data的问题，使用了两个生成器和两个判别器
  - 假设我们有一个图像翻译问题，需要把X域的图片映射到Y域，比如把莫奈风格的画作翻译成照片，这里的域可以理解为具有某种属性的集合。我们同时构造一个对偶问题，也就是把Y域映射到X域。从X到Y的映射记为G，从Y到X的映射记为F，G和F就是两个生成器。判别器DX用于区分X域的图片是来自真实图片还是生成图片，判别器DY类似。
  -Pix2Pix中一个问题是unpaired data没办法做L1 loss，L1 loss作为一个强引导关系，不仅能让模型加快收敛，而且还能避免模式崩溃问题，所以很重要。在CycleGAN中我们就构造一个L1 loss：x通过G转换到Y域，然后再通过F转换回X域，转换回来的x-hat和x做L1 loss
  这个时候就不容易发生模式崩溃问题，因为如果输入不同的x，都输出同一个y，那同一个y就没办法恢复成不同的x-hat，这样L1 loss就会很大
- 做法
  - 假设有两个域的数据，记为A，B。如A域是普通的马，B域是斑马。A->B的转换缺乏监督信息
  - 对于A域的所有图像，学习一个网络Generator A2B，生成B域的图像；对于B域的所有图像，学习一个网络Generator B2A，生成A域的图像
  - 通过A->fake_B->rec_A，以及B->fake_A->rec_B，可以设计重建损失。其中A->fake_B和fake_A->rec_B的网络是一模一样的
- 训练过程分两步：
  - 对于A域的某张图像，送入Generator A2B，生成fake_B；
  - 将fake_B送入Generator B2A，得到重构后的图像rec_A
  - 将重构后的图像rec_A和原图A做均方误差，实现了有监督的训练
- 网络结构
  - cycleGAN的生成器采用U-Net，判别器采用LSGAN
- loss
  - A域和B域的GAN loss，以及Cycle consistency loss
  - 公式：
  - 整个过程end to end训练，效果非常惊艳，利用这一框架可以完成非常多有趣的任务。
- 优缺点
  - 优点：用如此简单的模型，成功解决了图像翻译领域面临的数据缺乏问题。不需要配对的两个场景的相互映射，实现了图像间的相互转换，是图像翻译领域的又一重大突破。
### StarGAN
- 引入
  - 前面的Pix2Pix和CycleGAN都只能进行两个域之间的互相转换，如果要做多个域之间的互相转换，比如在人脸生气、开心、恐惧等不同表情之间转换，就需要训练N*(N-1)个模型。如何训练一个模型来实现多个域之间的互相转换呢?
- 关注问题：
  - cycleGAN需要针对每一对域训练一个模型，效率太低
  - cycleGAN对每一个域都需要搜集大量数据。以橘子转换为红苹果和青苹果为例，不管是红苹果还是青苹果，都是苹果，只是颜色不一样而已，这两个任务信息是可以共享的，没必要分别训练两个模型。
- 思路：
  - StarGAN解决多域转换的方法其实就是**cGAN + CycleGAN**
  - 对于判别器D：除了具有判断图片是否真实的功能外，还要有判断图片属于哪个类别的能力（这个能力在cGAN中是通过拼接学到的）
  - 对于生成器G：输入除了图片，还需要有目标领域信息，对应于cGAN中的标签信息。而和CycleGAN一样，通过生成器转换到目标域后，还要能再转换回来，做L1 loss。
- 优点
  - 提出多领域无监督图像翻译框架，实现了多个领域的图像转换
  - 不同领域的数据可以混合在一起训练，提高了数据利用率
## 文本生成
### GAN为什么不适合文本任务
### seqGAN用于文本生成
## others
### 数据增广
- 行人重识别
  - 难点：不同摄像头下拍摄的人物环境、角度差别非常大，导致承载较大的domain gap。
  - 解决方案：考虑使用cycleGAN来生成不同摄像头下的数据进行数据增广。对于每一对摄像头都训练一个cycleGAN，这样就可以实现将一个摄像头下的数据转换成另一个摄像头下的数据，但是内容（人物）保持不变。
  - paper
    - Zheng Z , Zheng L , Yang Y . Unlabeled Samples Generated by GAN Improve the Person Re-identification Baseline in Vitro[C]// 2017 IEEE International Conference on Computer Vision (ICCV). IEEE Computer Society, 2017.
    - Zheng, Z., Yang, X., Yu, Z., Zheng, L., Yang, Y., & Kautz, J. Joint discriminative and generative learning for person re-identification. IEEE Conference on Computer Vision and Pattern Recognition (CVPR)[C]. 2019.
### 图像超分辨与图像补全
- 作为图像翻译问题，训练一个端到端的网络，输入是原始图片，输出是超分辨率后的图片，或者是补全后的图片
- 超分辨率：
  - paper: Ledig C , Theis L , Huszar F , et al. Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network[J]. CVPR, 2016.
  - 思路：增加判别器，使得超分辨率模型输出的图片更加清晰，更符合人眼主观感受
- 图像补全
  - paper: Iizuka S , Simo-Serra E , Ishikawa H . Globally and locally consistent image completion[J]. ACM Transactions on Graphics, 2017, 36(4):1-14.
  - 思路：全局+局部一致性的GAN实现图像补全，使得修复后的图像不仅细节清晰，且具有整体一致性。
### 语音领域
- 音频去噪：SEGAN，缓解了传统方法支持噪声种类稀少，泛化能力不强的问题
- 语音增强：提高了ASR系统的识别率
## problem
- pix2pix中，L1 loss比L2 loss保边缘，以及L1 loss和L2 loss的先验问题怎么理解？
- cycleGAN中，DX和DY的输入有哪些？Cycle回来的算生成图吗？cycleGAN的loss包括哪些？