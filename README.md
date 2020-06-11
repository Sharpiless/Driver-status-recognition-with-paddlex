# 基于PaddleX的驾驶员状态识别和Paddle-Lite部署

![](https://ai-studio-static-online.cdn.bcebos.com/5c98a2186558456eb4eb014d85c866c75cc69bfcfe454eaf864c7e4b42a36a01)


## 项目简介：

该项目使用PaddleX提供的图像分类模型，在 kaggle 驾驶员状态检测数据集进行训练；

训练得到的模型能够区分驾驶员正常驾驶、打电话、喝水等等不同动作，准确率为0.979；

并使用PaddleLite进轻量级推理框架进行部署；

该项目使用CPU环境或GPU环境运行，PaddleX会自动选择合适的环境；

## 目录：

1. PaddleX工具简介；
2. 数据集简介；
3. 定义数据加载器；
4. 定义并训练模型；
5. 评估模型性能；
6. 使用PaddleLite进行模型部署；
7. 总结

## 一、PaddleX 工具简介：

![](https://ai-studio-static-online.cdn.bcebos.com/36ddce8a108641eaacdce72c0d43c6b841523961829d43929c6f9c45b8ebba8e)

**PaddleX简介**：PaddleX是飞桨全流程开发工具，集飞桨核心框架、模型库、工具及组件等深度学习开发所需全部能力于一身，打通深度学习开发全流程，并提供简明易懂的Python API，方便用户根据实际生产需求进行直接调用或二次开发，为开发者提供飞桨全流程开发的最佳实践。目前，该工具代码已开源于GitHub，同时可访问PaddleX在线使用文档，快速查阅读使用教程和API文档说明。

**PaddleX代码GitHub链接**：https://github.com/PaddlePaddle/PaddleX/tree/develop

**PaddleX文档链接**：https://paddlex.readthedocs.io/zh_CN/latest/index.html

**PaddleX官网链接**：https://www.paddlepaddle.org.cn/paddle/paddlex

![](https://ai-studio-static-online.cdn.bcebos.com/8c38d37fa83e44f7af5574ab395df925080903edab90485c8b2cb7ca7cf750dd)

### 二、数据集简介：

数据集地址：[https://www.kaggle.com/c/state-farm-distracted-driver-detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

![](https://storage.googleapis.com/kaggle-competitions/kaggle/5048/media/output_DEb8oT.gif)


该数据集由kaggle提供，共包括十个类别：

	    'c0': 'normal driving',

        'c1': 'texting-right',
        
        'c2': 'talking on the phone-right',
        
        'c3': 'texting-left',
        
        'c4': 'talking on the phone-left',
        
        'c5': 'operating the radio',
        
        'c6': 'drinking',
        
        'c7': 'reaching behind',
        
        'c8': 'hair and makeup',
        
        'c9': 'talking to passenger'


```python
# 解压数据集（注意要根据环境修改路径——
!unzip /home/aistudio/data/data35503//imgs.zip -d /home/aistudio/work/imgs
!cp /home/aistudio/data/data35503/lbls.csv /home/aistudio/work/
```

安装paddleX和1.7.0版本的paddlepaddle（这是由于paddlex并不支持最新版本）


```python
!pip install paddlex -i https://mirror.baidu.com/pypi/simple
!pip install paddlepaddle-gpu==1.7.0.post107 -i https://mirror.baidu.com/pypi/simple
```


```python
import os
# 设置使用0号GPU卡（如无GPU，执行此代码后仍然会使用CPU训练模型）
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.chdir('/home/aistudio/work/')
```


```python
# jupyter中使用paddlex需要设置matplotlib
import matplotlib
matplotlib.use('Agg') 
import paddlex as pdx
```

### 三、定义数据加载器：

这里主要是通过 pdx.datasets.ImageNet 类定义用于识别任务的数据加载器；


```python
import paddlehub as hub
import paddle.fluid as fluid
import numpy as np

base = './imgs/'

datas = []
for i in range(10):
    c_base = base+'train/c{}/'.format(i)
    for im in os.listdir(c_base):
        pt = os.path.join('train/c{}/'.format(i), im)
        line = '{} {}'.format(pt, i)
        # print(line)
        datas.append(line)

np.random.seed(10)
np.random.shuffle(datas)

total_num = len(datas)
train_num = int(0.8*total_num)
test_num = int(0.1*total_num)
valid_num = total_num - train_num - test_num

print('train:', train_num)
print('valid:', valid_num)
print('test:', test_num)

with open(base+'train_list.txt', 'w') as f:
    for v in datas[:train_num]:
        f.write(v+'\n')

with open(base+'test_list.txt', 'w') as f:
    for v in datas[-test_num:]:
        f.write(v+'\n')

with open(base+'val_list.txt', 'w') as f:
    for v in datas[train_num:-test_num]:
        f.write(v+'\n')

with open(base+'labels.txt', 'w') as f:
    for i in range(10):
        f.write('ch{}\n'.format(i))
```

    train: 17939
    valid: 2243
    test: 2242



```python
from paddlex.cls import transforms
train_transforms = transforms.Compose([
    transforms.RandomCrop(crop_size=224),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize()
])
eval_transforms = transforms.Compose([
    transforms.ResizeByShort(short_size=256),
    transforms.CenterCrop(crop_size=224),
    transforms.Normalize()
])
```


```python
train_dataset = pdx.datasets.ImageNet(
    data_dir='data',
    file_list='data/train_list.txt',
    label_list='data/labels.txt',
    transforms=train_transforms,
    shuffle=True)
eval_dataset = pdx.datasets.ImageNet(
    data_dir='data',
    file_list='data/val_list.txt',
    label_list='data/labels.txt',
    transforms=eval_transforms)
```

    2020-05-18 07:57:03 [INFO]	Starting to read file list from dataset...
    2020-05-18 07:57:03 [INFO]	17939 samples in file data/train_list.txt
    2020-05-18 07:57:03 [INFO]	Starting to read file list from dataset...
    2020-05-18 07:57:03 [INFO]	2243 samples in file data/val_list.txt



```python
num_classes = len(train_dataset.labels)
print(num_classes)
```

    10


## 四、定义并训练模型：

这里使用 MobileNetv3 进行训练；

MobileNetv3详细介绍可以看我的这一篇博客：

[https://blog.csdn.net/weixin_44936889/article/details/104243853](https://blog.csdn.net/weixin_44936889/article/details/104243853)

这里简单复述一下：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211100812235.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211101149861.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

琦玉老师 和 龙卷（~~阿姨~~）小姐姐 告诉我一个道理——画风越简单，实力越强悍；

这篇论文只有四个词，我只能说：不！简！单！

### MobileNet简介：

为了使深度学习神经网络能够用于移动和嵌入式设备，

MobileNet 提出了使用深度分离卷积减少参数的方法；

#### DW Conv：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020020920461233.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

即先将特征层的每个channel分开，然后分别做卷积，这样参数大约少了N倍（N是输出特征层的channel数）；

#### PW Conv：
就是1×1卷积，用于融合不同channel的特征；


### （一）论文地址：

[《Searching for MobileNet V3》](https://arxiv.org/abs/1905.02244?context=cs)

### （二）核心思想：
1. 使用了两个黑科技：NAS 和 NetAdapt 互补搜索技术，其中 NAS 负责搜索网络的模块化结构，NetAdapt 负责微调每一层的 channel 数，从而在延迟和准确性中达到一个平衡；
2. 提出了一个对于移动设备更适用的非线性函数 $h-swish[x]=x\frac{ReLU6(x+3)}{6}$；
3. 提出了 $MobileNetV3-Large$ 和 $MobileNetV3-Small$ 两个新的高效率网络；
4. 提出了一个新的高效分割（指像素级操作，如语义分割）的解码器（$decoder$）；

### （三）Platform-Aware NAS for Block-wise Search：
#### 3.1 MobileNetV3-Large：
对于有较大计算能力的平台，作者提出了 MobileNetV3-Large，并使用了跟 MnanNet-A1 相似的基于 RNN 控制器和分解分层搜索空间的 NAS 搜索方法；

#### 3.1 MobileNetV3-Small：
对于有计算能力受限制的平台，作者提出了 MobileNetV3-Small；

这里作者发现，原先的优化方法并不适用于小的网络，因此作者提出了改进方法；

用于近似帕累托最优解的多目标奖励函数定义如下：

$ACC(m)×[LAT(m)/TAR]^w$

其中 $m$  是第 $m$ 个模型的索引，$ACC$ 是模型的准确率，$LAT$ 是模型的延迟，$TAR$ 是目标延迟；

作者在这里将权重因数 $w=-0.07$ 改成了 $w=-0.15$，最后得到了一个期望的种子模型（initial seed model）；

### （四）NetAdapt for Layer-wise Search：
第二个黑科技就是 NetAdapt 搜索方法，用于微调上一步生成的种子模型；

NetAdapt 的基本方法是循环迭代以下步骤：

> 1. 生成一系列建议模型（proposals），每个建议模型代表了一种结构改进，满足延迟至少比上一步的模型减小了 $\delta$，其中 $\delta=0.01|L|$，$L$ 是种子模型的延迟；
> 2. 对于每一个建议模型，使用上一步的预训练模型，删除并随机初始化改进后丢失的权重，继续训练 $T$ 步来粗略估计建议模型的准确率，其中 $T=10000$；
> 3. 根据某种度量，选取最合适的建议模型，直到达到了目标延迟 $TAR$；

作者将度量方法改进为最小化（原文是最大化，感觉是笔误）：$\frac{\Delta Acc}{\Delta latency}$

其中建议模型的提取方法为：

> 1. 减小 Expansion Layer 的大小；
> 2. 同时减小 BottleNeck 模块中的前后残差项的 channel 数；

### （五）Efficient Mobile Building Blocks：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211114512875.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

作者在 BottleNet 的结构中加入了SE结构，并且放在了depthwise filter之后；

由于SE结构会消耗一定的计算时间，所以作者在含有SE的结构中，将 Expansion Layer 的 channel 数变为原来的1/4；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200209211456495.png)

其中 SE 模块首先对卷积得到的特征图进行 Squeeze 操作，得到特征图每个 channel 上的全局特征，

然后对全局特征进行 Excitation 操作，学习各个 channel 间的关系，

从而得到不同channel的权重，最后乘以原来的特征图得到最终的带有权重的特征；

### （六）Redesigning Expensive Layers：
作者在研究时发现，网络开头和结尾处的模块比较耗费计算能力，因此作者提出了改进这些模块的优化方法，从而在保证准确度不变的情况下减小延迟；

#### 6.1 Last Stage：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211113345896.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

在这里作者删掉了 Average pooling 前的一个逆瓶颈模块（包含三个层，用于提取高维特征），并在 Average pooling 之后加上了一个 1×1 卷积提取高维特征；

这样使用 Average pooling 将大小为 7×7 的特征图降维到 1×1 大小，再用 1×1 卷积提取特征，就减小了 7×7=49 倍的计算量，并且整体上减小了 11% 的运算时间；

#### 6.2 Initial Set of Filters：

之前的 MobileNet 模型开头使用的都是 32 组 3×3 大小的卷积核并使用 ReLU 或者 swish 函数作为激活函数；

作者在这里提出，可以使用 h-switch 函数作为激励函数，从而删掉多余的卷积核，使得初始的卷积核组数从 32 下降到了 16；

### （7）hard switch 函数：

之前有论文提出，可以使用 $swish$ 函数替代 ReLU 函数，并且能够提升准确率；

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211115954738.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)
其中 switch 函数定义为：

$swish[x]=x×\sigma(x)$，其中 $\sigma(x)=sigmoid(x)=1/（1+e^{-x}）$；

由于 sigmaoid 函数比较复杂，在嵌入式设备和移动设备计算消耗较大，作者提出了两个解决办法：

#### 7.1 h-swish 函数：

将 swish 中的 sigmoid 函数替换为一个线性函数，将其称为 h-swish：

$h$-$swish[x]=x\frac{ReLU6(x+3)}{6}$

![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211120639147.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

#### 7.2 going deeper：

作者发现 swish 函数的作用主要是在网络的较深层实现的，因此只需要在网络的第一层和后半段使用 h-swish 函数；

### （八）网络结构：

#### 8.1 MobileNetV3-Large：
![在这里插入图片描述](https://img-blog.csdnimg.cn/20200211120851338.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF,t_70)

#### 8.2 MobileNetV3-Small：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2020021112105278.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NDkzNjg4OQ==,size_16,color_FFFFFF



```python
# 定义并训练模型
model = pdx.cls.MobileNetV3_small_ssld(num_classes=num_classes)
model.train(num_epochs=2,
            train_dataset=train_dataset,
            train_batch_size=32,
            log_interval_steps=20,
            eval_dataset=eval_dataset,
            lr_decay_epochs=[1],
            save_interval_epochs=1,
            learning_rate=0.01,
            save_dir='output/mobilenetv3')
```

## 五、评估模型性能


```python
save_dir = 'output/mobilenetv3/best_model'
model = pdx.load_model(save_dir)
model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)
```

    2020-05-18 09:25:35 [INFO]	Model[MobileNetV3_small_ssld] loaded.
    2020-05-18 09:25:35 [INFO]	Start to evaluating(total_samples=2243, total_steps=2243)...


    100%|██████████| 2243/2243 [00:58<00:00, 38.38it/s]





    OrderedDict([('acc1', 0.9790459206419974), ('acc5', 1.0)])



## 六、使用PaddleLite进行模型轻量化部署

PaddleLite 是 paddle 提供的轻量级推理框架；

![](https://ai-studio-static-online.cdn.bcebos.com/6656c91353df46c38fbf935ee7f0ef8427a54462db694cca87f93e7c39c8716f)

文档地址：

[https://paddle-lite.readthedocs.io/zh/latest/index.html#](https://paddle-lite.readthedocs.io/zh/latest/index.html#)

简介：
Paddle-Lite 框架是 PaddleMobile 新一代架构，重点支持移动端推理预测，特点为高性能、多硬件、轻量级 。

支持PaddleFluid/TensorFlow/Caffe/ONNX模型的推理部署，目前已经支持 ARM CPU, Mali GPU, Adreno GPU, Huawei NPU 等多种硬件，

正在逐步增加 X86 CPU, Nvidia GPU 等多款硬件，相关硬件性能业内领先。



```python
# 进行模型量化并保存量化模型
pdx.slim.export_quant_model(model, eval_dataset, save_dir='./quant_mobilenet')
print('done.')
```


```python
# 加载模型并进行评估
quant_model = pdx.load_model('./quant_mobilenet')
quant_model.evaluate(eval_dataset, batch_size=1, epoch_id=None, return_details=False)
```

## 七、总结：

### 在本项目中我们完成了以下任务：

1. 使用PaddleX在驾驶员状态识别数据集训练了MobileNetv3模型；

2. 使用PaddleLite实现了模型的轻量化部署；

## 关于作者：
> 北京理工大学 大二在读

> 感兴趣的方向为：目标检测、人脸识别、EEG识别等

> 将会定期分享一些小项目，感兴趣的朋友可以互相关注一下：[主页链接](http://aistudio.baidu.com/aistudio/personalcenter/thirdview/67156)

> 也欢迎大家fork、评论交流

> 作者博客主页：[https://blog.csdn.net/weixin_44936889](https://blog.csdn.net/weixin_44936889)
