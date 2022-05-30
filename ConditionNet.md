## 要搞清楚的问题：

1. ConditionNet的输入输出分别是什么
2. 如何训练ConditionNet
3. ConditionNet的训练集和测试集？
4. ConditionNet的输出是怎么作用到Basenet的

## 训练过程

1. ConditionNet的输入：一组退化模式相同的图像(称为support set)

   输出：描述退化特征的向量

   ![image-20220527124339099](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205281030418.png)

2. 输入多组退化模式的图像，与整个模型一起训练，而并非单独训练ConditionNet

3. 训练集：多组不同退化模式的图像( shape= (k,n*3,h,w), k表示退化模式种类数(任务数)，n代表每个退化模式的图片数，称为support set)

   测试集：shape= (k,3,h,w)提取要超分图像的特征

4. ***BaseNet multiplies its convolution weights with modulated conditional features in channel-wise***

   使用方法: 将卷积核的权重与提取出的退化特征逐通道相乘

   哪层卷积核权重？

## 代码阅读

EqualLinear：自定义的线性层

ConditionNet：ConditionNet，输出为Condition_weight

sr_net：Generator网络主干网，forward输入为LR图片和ConditionNet输出的权重，推理由Backbone网络Resblock进行

Resblock：由BaseConv2d组成

BaseConv2d:将卷积核权重和Condition_feature系数相乘

Modulation：

退化特征信息传递过程：

SupportSet[task_size, support_size*3, h, w] ---ConditionNet---> 

conditional_feature(代表SupportSet的退化特征，传入sr_net)[task_size, 128, 1, 1] ---Modulation[n_block, n_conv_each_block, task_size, 1, 64, 1, 1], repeat_interleave[n_block, n_conv_each_block, batch, 1, 64, 1, 1]---> 

conditional_feature(使得退化特征能传入backbone网络，传入ResBlock)[n_conv_each_block, batch, 1, 64, 1, 1] -----> condition_feature作用，改变BaseConv2d卷积核权重



## 考虑可行性

1. 代码

   应该是可行的，在网络中加入一个conditionNet那样的结构就可以了

   Generator的LOSS：

   real-esrgan先用L1-loss训练了一次，再用L1-loss，perceptual-loss，GAN-loss

   ConditionNet要用contrastive loss $L_{con}+\lambda\times L_{res}(衡量重建质量的loss)$

2. 数据集

   需要的数据集：LR->HR的训练对以及输入ConditionNet的SupportSet

   主要是这个SupportSet的要求是什么, 多种退化模型生成的图片按类分好

## 开始动手

1. 从哪个代码作为基石开始改？

   鉴于代码有效性，从real-ESRGAN代码开始

2. ConditionNet COPY过来

3. 找到训练的地方(看Training.md)

   1. 生成多尺度图像，剪裁sub-image(均可选)
   2. 存储训练图片路径的txt
   3. 
   4. 运行realesrgan/train.py

4. 更改对应部分：

   - [ ] 计算loss
   
   - [x] 更改网络结构
   
   - [ ] 网络参数调整（计算维度等）
   
   - [ ] 从原数据得到support_set
   
     1. 整个数据集里相同的退化类型图片作为输入
   
     - [ ] 创建每个退化类的support_set对应图片的memory pool
     - [ ] 标记每张图的退化类型
   
     
   
     2. 同一批次的退化类型图片作为输入(batchsize=12)
     3. [ ] 直接将当前批次LR作为ConditionNet的输入
     4. [ ] 12142 64950