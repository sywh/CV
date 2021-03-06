# evaluation
## subjective
- 主观评价需要花费大量人力物力
- 主观评价带有主观色彩，且有些badcase没看到容易造成误判
- 如果GAN过拟合，那么生成的样本会非常真实，人类主观评价得分会非常高，可这并不是一个好的GAN（是指多样性不够吗？）
## inception score
- 公式：
- 理解：对于一个在ImageNet上训练良好的GAN，其生成的样本丢给Inception网络进行测试时，得到的判别概率应该具有如下特性：
  - 对同一个类别的图片，其输出的概率分布应该趋向于一个脉冲分布，可以保证生成样本的准确性
  - 对于所有类别，其输出的概率分布应该趋向于一个均匀分布，这样才不会出现mode dropping等，可以保证生成样本的多样性
  - 实际实验表明，IS和人的主观判别趋向一致
  - 一个训练良好的GAN, p(y|x)接近脉冲分布，p(y)趋近于均匀分布，两者KL散度会很大，Inception Score自然就高。
  - IS的计算没有永达真实数据
- 特点：可以一定程度上衡量生成样本的多样性和准确性，但是无法检测过拟合（GAN的过拟合指什么？）。mode score也是如此。不推荐在和ImageNet数据集差别比较大的数据上使用。
## mode score
- 公式
- 理解：作为inception score的改进版本，添加了关于生成样本和真实样本预测概率分布相似性度量一项
## kernel MMD(Maximum Mean Discrepancy)
- 推荐了解
## wasserstein distance
- 公式：
- 理解：
  - wasserstein distance在最优传输问题中通常叫做推土机距离
  - wasserstein distance可以衡量两个分布之间的相似性。距离越小，分布越相似。
- 特定：如果特征空间选择合适，会有一定的效果。但是计算复杂度为O(n^3)太高
## Frechet Inception Distance(FID)
- 公式：
- 理解：
  - FID距离计算真实样本，生活样本在特征空间上的距离
  - 首先利用Inception网络提取特征，然后使用高斯模型对特征空间进行建模。根据高斯模型的均值和协方差来进行距离计算
- 特点：
  - 尽管只计算了特征空间的前两阶矩，但是鲁棒，且计算高效
## 1-Nearest Neighbor classifier
- 使用留一法，结合1-NN分类器（别的也行）计算真实图片，生成图片的精度。如果两者接近，则精度接近50%，否则接近0%
- 对于GAN的评价问题，作者分别用正样本的分类精度，生成样本的分类精度来衡量生成样本的真实性，多样性
  - 对于真实样本xr，进行1-NN分类的时候，如果生成的样本越真实，则真实样本空间R将被生成样本xg包围，那么xr的精度会很低。
  - 对于生成样本xg，进行1-NN分类的时候，如果生成的样本多样性不足，由于生成的样本聚在几个mode，则xg就很容易和xr区分，导致精度会很高。
- 特点：理想的度量指标，可以检测过拟合。
## others
- AIS，KDE方法也可以用于评价GAN，但这些方法不是model agnostic metrics。也就是说，这些评价指标的计算无法只利用：生成样本和真实样本来计算。
