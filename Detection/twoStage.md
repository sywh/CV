# twoStage
## RCNN
- 创新点
  - 使用CNN对Region Proposals计算feature vector。从经验驱动特征(SIFT, HOG) 到数据驱动特征(CNN feature map)，提高特征对样本的表示能力
  - **采用大样本下(ILSVRC)有监督预训练和小样本(PASCAL)微调(finetune)的方法解决小样本难以训练甚至过拟合的问题**
- 步骤
  - 预训练并微调：在warped的VOC数据集上微调（全连接层还是所有层？微调的损失函数是什么？）
  - 利用选择性搜索(Selective Search)算法提取约2000个proposals，resize到227×227大小
  - 计算CNN特征(4096维特征向量)，并保存到本地磁盘
  - 训练SVM：每个类训练一个二分类SVM
  - 边界框回归(Bounding boxes regression)
- 性能
  - 精度：当时SOTA，VOC2007测试集mAP=58.5%
## FastRCNN
- 创新点
  - **只对整副图像进行一次特征提取，避免RCNN中的冗余特征提取**
  - **用RoIPooling代替resize，提取Region Proposal特征**
  - 网络末尾采用并行的两个分支，分别输出分类结果和边界框回归结果，除Region Proposal以外，实现了端到端的多任务训练，也不需要额外的特征存储空间(RCNN特征需要保存到本地)
  - 采用SVD对网络末尾并行的全连接层进行分解，减少计算复杂度，加快检测速度
- RoIPooling
  - 作用：
    - 将不同大小的RoI转换为固定大小，是针对RoI的Pooling，其特点是输入特征图尺寸不固定，但是输出特征图尺寸固定。**RoI是指候选框在feature map上的位置**
    - **最大的好处在于极大地提高了处理速度**（为什么RoIPooling能提高处理速度？）
  - 输入
    - 特征图：在FasterRCNN中，是与RPN共享的那个特征图
    - RoIs，表示所有RoI的N×5的矩阵，N表示RoI的数量，第一列表示图像index，其余四列表示坐标。**值得注意的是，坐标的参考系不是针对feature map这张图的，而是针对原图的**
  - 操作
    - 将区域分为相同大小的sections(sections数量和输出的维度相同)
    - 对每个sections进行maxPooling操作
- 性能
  - 速度：2.3s，其中2s用于生成2000个ROI
- 缺点
  - 依赖于外部候选区域算法，如选择性搜索，这些算法在CPU上运行且速度很慢
## FasterRCNN
- 创新点
  - **采用候选区域网络(RPN)代替选择性搜索，在生成RoI时效率更高，每幅图像10ms**
- RPN
  - 输入为feature map，输出4k个坐标预测边界框，和2k个objectness度量边界框是否包含目标（为什么是2k个？为什么不能用1k个？）
  - 对特征图的每一个位置，RPN会做k次预测
  - **RPN不直接预测边界框位置，而是预测边界框相对锚框的偏移量**（RPN怎么训练，不需要NMS吗？）
  - 锚框是精心挑选的，具有不同比例和宽高比，可使早期训练更加稳定和简便
## R-FCN
- 创新点：
  - 使用**位置敏感得分图(position-sensitive score maps)**解决分类平移不变性和检测平移差异性之间的难题
- 性能
  - 精度：ResNet-101+R-FCN, VOC2007 测试集上83.6%mAP
## FPN
- 创新点
  - 通过特征金字塔网络，改变网络连接，在基本不增加原有模型计算量的情况下，解决目标检测中多尺度问题，大幅度提升小物体(small object)检测的性能
- 针对问题：多尺度物体检测
  - 传统方法
    - **图像金字塔(image pyramid)**：即多尺度训练和测试，但该方法计算量大，耗时较久
    - **特征分层**：每层分别预测对应scale分辨率的检测结果，如SSD算法。该方法强行让不同层学习同样的语义信息，但实际上不同深度对应于不同层次的语义特征，浅层网络分辨率高，学到更多是细节特征；深层网络分辨率低，学到更多是语义特征
  - 面临挑战
    - 如何学习具有强语义信息的多尺度特征表示？
    - 如何设计通用的特征表示来解决物体检测中的多个子问题？如object proposal, box localization, instance segmentation
    - 如何高效计算多尺度的特征表示？
- FPN算法
  - Bottom-up pathway(自底向上线路)
    - 下：指low-level，对应于提取的低级语义(浅层)特征；上：指high-level，对应于提取的高级语义(高层)特征
    - feature pyramid: 一些尺度(scale)因子为2，后一层feature map大小是前一层feature map大小的二分之一，根据此关系构成了feature pyramid; 然后还有很多层输出feature map是一样的大小，作者将这些层归为同一stage，为每个stage定义一个pyramid level; 作者将每个stage的最后一层输出作为feature map，构成feature pyramid
  - Lateral connections(横向连接)
    - 将上采样的结果和bottom-up pathway生成的相同大小的feature map进行融合(element-wise addition)
    - **bottom-up feature要先经过1×1卷积层，目的是为了减少通道维度**
  - Top-down pathway(自顶向下线路)
    - 上采样过程
  - 其它
    - FPN主要应用在FasterRCNN中的RPN和FastRCNN(用于object detection)两个模块中
    - **RPN和FastRCNN分别关注的是召回率和精确率，对比的指标分别为Average Recall(AR)和Average Prevision(AP)**
- 优点
  - **通过横向连接，每一层预测所用的feature map都融合了不同分辨率，不同语义强度的特征，融合的不同分辨率的feature map分别做对应分辨率大小的物体检测，这样保证了每一层都有合适的分辨率以及强语义特征**(也就是说融合的feature map负责两种分辨率大小的物体检测？)
  - 只在原网络基础上增加额外的跨层连接，在实际应用中几乎不增加额外的时间和计算量
## MaskRCNN
- 创新点
  - Backbone: ResNeXt-101+FPN
  - **用RoIAlign替换RoIPooling**
- 步骤
  - 将预处理后图片输入一个预训练好的神经网络(如ResNext)中，获得对应的feature map
  - 对这个feature map中每一点预设给定RoI，从而获得多个候选RoI
  - 将这些候选RoI送入RPN网络进行二值分类(前景或背景)和BB回归，过滤掉一部分候选的RoI
  - 对剩下的RoI进行RoI Align操作
  - **对这些RoI进行分类(N类别)，BB回归和Mask生成(在每一个RoI里面进行FCN操作)**
- RoIAlign
  - RoIPooling操作中两次量化造成区域不匹配(mis-alignment)问题
  - 预选框位置通常都是由模型回归得到，一般是浮点数，而池化后的特征图要求尺寸固定，故RoIPooling操作存在两次量化过程
    - 将候选框边界量化为整数
    - 将量化后的边界区域平均分割成k*k个单元(bin)，对每一个单元的边界进行量化
    **经过上述两次量化，此时候选框已经和回归出来的位置有一定偏差，会影响检测或者分割的准确度。作者把它总结为不匹配问题(misalignment)。小目标受misalignment问题的影响更大**
  - RoI的思路
    - 取消量化操作，使用双线性内插的方法获得坐标为浮点数的像素点上的图像数值
  - 步骤
    - 遍历每一个候选区域，保持浮点数边界不做量化
    - 将候选区域分割成k*k个单元，每个单元的边界也不做量化
    - **在每个单元中计算固定的四个坐标位置，用双线性内插方法计算出这四个位置的值，然后进行最大池化操作**
## DetNet
## CBNet