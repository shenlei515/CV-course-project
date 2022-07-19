## 简介

经典建模成多种退化模型的叠加，然而由于现实情况的复杂，往往具体的退化过程是不能确定的，这类方法被称为直接建模，因此我们考虑用非直接建模的方法处理这类退化，具体而言就是利用学习到的数据分布来获得退化模型，但这种方法也受限于训练集中的退化情况，其泛化表现一般。因此我们结合这两者的

## 主要贡献

1. 用高阶退化模型来建模退化过程，用sinc函数建模振铃，减小了振铃效应
2. 必要的网络调整
3. 可以应用于纯合成数据

## 经典退化模型

![img](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205252220080.png)

高斯模糊：DIP讲过不重复了，广义高斯核(高斯分布的阶数变化)

噪声模型：加性高斯噪声，泊松噪声(常用来建模传感器的统计误差)

上采样下采样：最近邻内插，区域剪裁，双线性内插，论文里说由于最近邻内插效果不好就没用

jpeg压缩退化：DIP讲过



## 二阶模型

一阶模型：即由上面提到的经典退化模型得到的组合模型，但是每个退化操作只会执行一次，由于现实场景比较复杂，这种简单的退化模型是不足以建模现实场景的

![img](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205252220574.png)

![img](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205252220838.png)



震荡多为带通滤波产生的效应，采用时域sinc滤波器来模拟频域中的理想滤波器

![image-20220525165516460](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205251655494.png)

最后一层JEPG退化和振铃响应的顺序会随机交换



**退化模型的作用是产生训练数据（实际上有的都是高清图片，模拟高清图片在退化作用下产生的低分辨率图片，低分辨率图片——高分辨率图片组成训练对）**

## 整体模型(Real-ESRGAN)

#### Generator部分

他们的模型大概就是在ESRGAN的基础上进行了预处理，通过pixel unshuffle将不同尺度(x2 x1)的压缩到X4的分辨率上，一起传到ESRGAN里进行训练

![image-20220525170102944](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205251701055.png)

像素重整(pixel unshuffle)：像素打乱(一种上采样方法，大概是通过卷积提取$r^2$个通道的特征图，再用周期筛选展平成一个$rH \times rW$)的逆操作

![在这里插入图片描述](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205251704104.jpeg)

RRDB block具体是什么？(需要看ESRGAN论文)

#### Discriminator部分

基本架构：带跳转连接(skip connection，就是残差链接)的U-net

![image-20220525173051562](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205251730622.png)

该U-net会输出每个像素的真实程度评分(图中看起来像频谱图一样的图就是真实评分图)-》原ESRGAN的Discriminator输出的是

![image-20220525192428188](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205251924232.png)

输出两个输入的真实度对比，如果左边为真实图片的概率>右边为真实图片的概率，输出1，反之输出0

运用了spectral normalization(具体是什么我也不知道)

#### 训练过程

先用L1 loss训练一个以增大PSNR(峰值信噪比)为导向的模型，作为generator的预训练模型---> 在这个模型基础上，用一个综合L1 loss、perceptual loss、GAN loss的损失函数来训练整个GAN

训练中用到一个将ground-truth图片进行锐化处理来锐化生成图像而不产生振铃现象(直接对生成图像锐化会产生振铃)，用了这个技巧的模型称为Real-ESRGAN+

## 蚀灼实验(去掉模型中的某一个部分来探究模型哪一个部分在发挥作用)

二阶退化模型->经典退化模型：不能很好的去除噪声和恢复模糊区域

去除sinc滤波器：振铃现象

U-net：通过残差链接可以改善本地细节，但也有可能出现部分不自然的纹理和增加训练的不稳定性，spectral normalization可以增加稳定性的同时改善图片纹理

更加复杂的模糊核(blur kernel)：大多数图像上效果和高斯核差不多，小部分有提升

## 待改进处

但是论文也指出了这种改进的高阶模型也并非完美，事实上它仅仅通过利用数据分析来拓展可以解决的退化问题范围，它有以下几种不足：

1. 扭曲的线
2. GAN训练产生的意料外的震荡
3. 无法建模没有见到过的未知降级



## 代码中的退化模型实现：

realesrgan/models/realesrgan_model.py

![image-20220525225250798](https://raw.githubusercontent.com/shenlei515/pics/main/img/202205252252958.png)