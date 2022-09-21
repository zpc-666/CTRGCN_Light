# 一、简介
<font size="3" color="black" face ="幼圆">本repo从[AI Studio项目](http://)迁移而来，移步[AI Studio项目](http://)fork运行最佳。</font>
## 1.1 任务要求
<font size="3" color="black" face ="幼圆">本次任务是要对[CTRGCN模型](https://arxiv.org/abs/2107.12213)进行轻量化，官方释放的在NTU-RGB+D数据集，joint模态，X-sub评测标准下的CTRGCN模型的Top1精度为`89.93%`，推理文件夹大小约为`14.5MB`（CTRGCN_joint.pdiparams：5.6MB，CTRGCN_joint.pdiparams.info：62.1KB，CTRGCN_joint.pdmodel：8.8MB），要求是：
1. 对模型进行轻量化，在inference模型小于`10M`条件下，在NTU-RGB+D数据集，joint模态，X-sub评测标准，Top1指标大于`89%`且最优指标取胜；
2. 完成轻量化后的模型的TIPC基础链条功能验证；
3. 复现后合入PaddleVideo
    
</font>

## 1.2 CTRGCN模型介绍
<font size="3" color="black" face ="幼圆">我们首先简要介绍CTRGCN的创新之处CTR-GC，即其空间建模部分，然后介绍CTR-GCN的整体结构。</font>
### 1.2.1 CTR-GC
![](https://ai-studio-static-online.cdn.bcebos.com/8ad48bfe3816479e8a5f23e0a30f0315079670f38beb4d2aaa13ffbcf4ce1e36)

<font size="3" color="black" face ="幼圆">在上图展示了CTR-GC的总体框架。结合上图和下图(b)，首先将输入特征X(形状为(B, T, N, C))利用特征变换函数T(·)进行特征变换（即降维），得到x1, x2(形状为(B, T, N, C//r))；然后分别进行时间维度全局池化得到x1, x2(形状为(B, N, C//r))，再利用通道相关性建模函数M(·)对x1, x2进行处理，M(·)可以是简单的act(x1-x2)，也可以是MLP(x1, x2)，论文里取的是act(x1-x2)，即x1(B, N, 1, C//r)-x2(B,  1, N, C//r)，这样得到通道特定的关系A’(形状为(B, N, N, C//r))，然后再利用1x1 conv进行升维操作，得到A’(形状为(B, N, N, C’))；最后和初始化为物理图拓扑结构的通道共享自适应图拓扑矩阵A(形状为(N, N))加权求和，得到通道细化的图拓扑结构(形状为(B, N, N, C’))，与使用1x1 conv特征变换后的特征X’ (形状为(B, T, N, C’)) 相乘进行空间信息聚集。
    
其中B表示一个batch的样本数量，T表示时间维度的帧数，N表示关节点数目，C表示特征维度，r表示维度降低的倍数，论文里取r=8；T(·)一般采用1x1 conv，act(·)表示激活函数，论文里选用Tanh(·)函数，MLP表示多层感知机，C’表示变换后（比如升维操作）的特征维度。,</font>

![](https://ai-studio-static-online.cdn.bcebos.com/5c6ff2654fdb4e368782e476d4045b29e7e64e7c11a148988e8d3c83d2577a12)

### 1.2.2 CTR-GCN整体结构
<font size="3" color="black" face ="幼圆">如上图(a)所示，CTR-GCN的基础Block是由空间建模模块和时序建模模块两部分组成。其中空间建模模块利用三个CTR-GC并行提取人体关键点之间的关系，将三个分支的结果求和作为最终的输出，然后经过常用的BN、ReLU层，并利用残差连接将输出和输入求和，作为时序建模模块的输入。为了提高模型速度，时序建模模块只使用了4个分支来建模多尺度的时域信息，提高了时域建模能力。每一个分支都包含一个1x1 conv进行通道降维。前三个分支包含两个不同空洞率的时序建模卷积和一个时间维最大池化。最后将四个分支的结果沿通道维拼接作为最终的输出。整个模型由十个上述描述的基础Block、一个全局平均池化层和一个全连接分类器组成。十个Block的输出通道数量是64-64-64-64-128-128-128-256-256-256。在第5个Block和第8个Block，时序建模卷积的stride=2，时间维度减半。</font>
    

# 二、轻量化模型介绍及其精度、大小
<font size="3" color="black" face ="幼圆">最初我们是一味地使用轻量的模块如ghost模块来代替原始卷积从而减少总的参数量和浮点数运算量，但无疑增加了模型的复杂性，模型结构变得更加琐碎，这使得模型权重文件（即CTRGCN_joint.pdiparams）得以大幅度减小，但带来的代价是模型结构文件大小（即CTRGCN_joint.pdmodel）增大，二者互相抵消，导致很难达到总的文件大小不超过10MB。在逐步替代CTRGCN的时序建模模块（即MultiScale_TemporalConv模块）和空间建模模块（即unit_gcn模块）时，我们发现CTRGCN的空间建模模块对导出的推理模型的权重文件大小（即CTRGCN_joint.pdiparams）及模型结构文件大小（即CTRGCN_joint.pdmodel）影响很大。同时，我们认为每个bock均使用提取人体关键点之间的关系的空间建模模块很可能是过度冗余的，即没必要在每个基础Block中均使用空间建模模块。因此，为了保持相同的时域感受野，我们保留了相同深度的时序建模模块，而是减少不必要的一些空间建模模块。左图为原始的CTRGCN基础块堆叠结构，右图是我们的轻量化的CTRGCN的基础块堆叠结构。

![](https://ai-studio-static-online.cdn.bcebos.com/93867b4386d14175b6ef4f1fb7a929b5e2219ae2a08444c2853f7e5fce9d162e)![](https://ai-studio-static-online.cdn.bcebos.com/7a17a581d8254604b33ac3d2b3dd1d6150084388e9f144e088fc2040b7eaeb9d)

根据输出时域分辨率，我们姑且将CTRGCN模型分为3个阶段（l1, l2, l3, l4为第一阶段，l5, l6, l7为第二阶段，l8, l9, l10为第三阶段），可以看到，我们在每个阶段的最后一个block保留了空间建模模块，除了第一个阶段是开始的第一个block保留了空间建模模块，其余阶段均是第二个block保留了空间建模模块，好处是在尽可能保留模型表达能力的同时，使用下采样后的分辨率来减少推理模型的总大小，其中TCN_unit是TCN_GCN_unit去掉空间建模模块的结构。除此之外，为了满足推理模型大小目标，我们进一步将base channel的大小从64调整为60。轻量化的CTRGCN模型即CTRGCN_light见PaddleVideo-develop/paddlevideo/modeling/backbones/ctrgcn_light.py
 
相同训练配置下轻量化CTRGCN与原始CTRGCN的精度及推理模型大小对比：


| 模型 | 精度 | 推理模型大小 |
| -------- | -------- | -------- |
| CTRGCN  | 89.93%     | 约14.5MB     |
| CTRGCN_light | 89.313% | 约9.7MB     |
 
 其中9.7MB = CTRGCN_joint.pdiparams：3.9MB，CTRGCN_joint.pdiparams.info：46KB，CTRGCN_joint.pdmodel：5.7MB，推理文件可以从这里[下载](https://aistudio.baidu.com/aistudio/datasetdetail/169741)
    
训练日志也可以从这里[下载](https://aistudio.baidu.com/aistudio/datasetdetail/169741)。visualdl下的训练日志曲线如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/ee6df541d85a430d9f36f29861110fb19404a0ca4d8b47a5b28c61e13e0e1cf4)


    


# 三、准备数据
<font size="3" color="black" face ="幼圆">我们是要在NTU-RGB+D数据集、joint模态、X-sub评测标准下对我们的轻量化的CTRGCN模型进行训练、验证及推理。CTRGCN的数据准备详情见PaddleVideo-develop/docs/zh-CN/dataset/ntu-rgbd.md，我们并没有完全按照其流程，即省略了数据集下载过程，因为**sh data/ntu-rgb-d/download_dataset.sh**命令需要翻墙下载，是不可行的，我们直接挂载大佬[小吴今天不熬夜](https://aistudio.baidu.com/aistudio/datasetdetail/146482)已经上传的下载好的数据集nturgbd_skeletons_s001_to_s017.zip，并使用**wget https://videotag.bj.bcebos.com/Data/statistics.zip** 命令得到statistics文件夹，之后的数据处理过程是与ntu-rgbd.md文档一致的，最终得到如下文件树形式

```txt
─── /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d
    ├── download_dataset.sh
    ├── nturgb+d_skeletons
    │   ├── S001C001P001R001A001.skeleton
    │   ├── S001C001P001R001A002.skeleton
    │   ├── S001C001P001R001A003.skeleton
    │   ├── S001C001P001R001A004.skeleton
    │   ├── S001C001P001R001A005.skeleton
    │   ├── S001C001P001R001A006.skeleton
    │   ├── S001C001P001R001A007.skeleton
    │   ├── ....
    │   └── S017C003P020R002A060.skeleton
    ├── denoised_data
    │   ├── actors_info
    │   │   ├── S001C001P001R001A024.txt
    │   │   ├── S001C001P001R001A025.txt
    │   │   ├── S001C001P001R001A026.txt
    │   │   ├── ....
    │   │   ├── S017C003P020R002A059.txt
    │   │   └── S017C003P020R002A060.txt
    │   ├── denoised_failed_1.log
    │   ├── denoised_failed_2.log
    │   ├── frames_cnt.txt
    │   ├── missing_skes_1.log
    │   ├── missing_skes_2.log
    │   ├── missing_skes.log
    │   ├── noise_length.log
    │   ├── noise_motion.log
    │   ├── noise_spread.log
    │   ├── raw_denoised_colors.pkl
    │   ├── raw_denoised_joints.pkl
    │   └── rgb+ske
    ├── raw_data
    │   ├── frames_cnt.txt
    │   ├── frames_drop.log
    │   ├── frames_drop_skes.pkl
    │   └── raw_skes_data.pkl
    ├── get_raw_denoised_data.py
    ├── get_raw_skes_data.py
    ├── seq_transformation.py
    ├── statistics
    │   ├── camera.txt
    │   ├── label.txt
    │   ├── performer.txt
    │   ├── replication.txt
    │   ├── setup.txt
    │   └── skes_available_name.txt
    ├── xview
    │   ├── train_data.npy
    │   ├── train_label.pkl
    │   ├── val_data.npy
    │   └── val_label.pkl
    └── xsub
        ├── train_data.npy
        ├── train_label.pkl
        ├── val_data.npy
        └── val_label.pkl
```

> 注：文件夹`denoised_data`、`raw_data`和`nturgb+d_skeletons`都为处理处理的临时文件，可在提取出`xview`和`xsub`后删除。

## 3.1 数据集下载及解压


```python
# 解压挂载好的数据集，并复制到PaddleVideo/data/ntu-rgb-d/下
!unzip -qo /home/aistudio/data/data146482/nturgbd_skeletons_s001_to_s017.zip -d /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d/
# 下载并解压得到statistics文件夹
%cd /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d
!wget https://videotag.bj.bcebos.com/Data/statistics.zip
!mkdir statistics
!unzip -qo statistics.zip -d statistics/ && rm -rf statistics.zip
%cd /home/aistudio/
```

## 3.2 数据集处理
<font size="3" color="black" face ="幼圆">处理过程要等待一段时间


```python
%cd /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d
# Get skeleton of each performer
!python get_raw_skes_data.py
# Remove the bad skeleton
!python get_raw_denoised_data.py
# Transform the skeleton to the center of the first frame
!python seq_transformation.py
%cd /home/aistudio/
```


```python
%cd /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d
!rm -rf nturgb+d_skeletons/
!rm -rf denoised_data/
!rm -rf raw_data/
%cd /home/aistudio/
```

    /home/aistudio/PaddleVideo-develop/data/ntu-rgb-d
    /home/aistudio


# 四、开始使用
## 4.1 环境依赖
<font size="3" color="black" face ="幼圆">本项目是完全基于AIStudio的32GB V100环境实现训练及验证、推理的
### 4.1.1 PaddleVideo套件的引入
<font size="3" color="black" face ="幼圆">我们的轻量化CTRGCN的训练、验证及推理全流程均基于开源套件PaddleVideo。


```python
#!git clone https://github.com/PaddlePaddle/PaddleVideo.git
```

### 4.1.2 安装所需依赖
<font size="3" color="black" face ="幼圆">需要安装PaddleVideo需要的依赖


```python
%cd /home/aistudio/PaddleVideo-develop
!pip install -r requirements.txt
%cd /home/aistudio/
```

## 4.2 模型训练
<font size="3" color="black" face ="幼圆">我们的轻量化的CTRGCN模型是完全按照CTRGCN的训练配置进行训练，仅修改了骨干网络名字为"CTRGCN_light"和修改base_channel为60，具体参数可见PaddleVideo-develop/configs/recognition/ctrgcn/ctrgcn_light_ntucs_joint.yaml
    
**对PaddleVideo-develop/paddlevideo/solver/custom_lr.py修改，将CustomWarmupAdjustDecay的输入参数num_iters默认为1，不然会报错：TypeError: unsupported operand type(s) for /: 'int' and 'NoneType'**


```python
# 运行该指令即可启动训练
%cd /home/aistudio/PaddleVideo-develop/
!python main.py --validate -c configs/recognition/ctrgcn/ctrgcn_light_ntucs_joint.yaml --seed 1
%cd /home/aistudio/
```

<font size="3" color="black" face ="幼圆">一个epoch的训练日志如下图：

![](https://ai-studio-static-online.cdn.bcebos.com/271b1bc90d4340ef81f8213930d2de65cad1f251b1674450a6309c2d6bde09b1)

## 4.3 模型评估
<font size="3" color="black" face ="幼圆">我们的训练好的轻量化CTRGCN模型权重文件可以从这里[下载](https://aistudio.baidu.com/aistudio/datasetdetail/169741)，训练日志在该文件夹下的logs文件夹中，可以通过visuadl可视化打开。你可以通过-w指定自己训练的模型权重。我们的训练好的模型精度为89.313%，大于89%。评估结果如下图所示：

![](https://ai-studio-static-online.cdn.bcebos.com/fdaabb964cd0450bb338e2117e0e73b121f69bc0b1c140d18089136185e2a8f9)
    


```python
# 运行该指令即可启动评估
%cd /home/aistudio/PaddleVideo-develop/
!python3.7 main.py --test -c configs/recognition/ctrgcn/ctrgcn_light_ntucs_joint.yaml -w /home/aistudio/PaddleVideo-develop/output/CTRGCN_light_my/CTRGCN_joint_best.pdparams
%cd /home/aistudio/
```

## 4.4 导出推理模型
<font size="3" color="black" face ="幼圆">我们利用训练好的轻量化CTRGCN模型权重文件导出其推理模型，得到的三个文件的总大小约为9.7MB，满足不超过10MB要求，可以从这里[下载](https://aistudio.baidu.com/aistudio/datasetdetail/169741)。


```python
# 运行该指令即可启动推理模型导出（出现的警告可以忽视）
%cd /home/aistudio/PaddleVideo-develop/
!python3.7 tools/export_model.py -c configs/recognition/ctrgcn/ctrgcn_light_ntucs_joint.yaml \
                                -p output/CTRGCN_light_my/CTRGCN_joint_best.pdparams \
                                -o inference/CTRGCN_light_joint
%cd /home/aistudio/
```

## 4.5 模型推理
<font size="3" color="black" face ="幼圆">我们使用轻量化CTRGCN导出的推理模型对`data/example_NTU-RGB-D_sketeton.npy`进行预测，输出的top1类别id为`58`，置信度为0.3077475428581238。我们使用官方预训练的CTRGCN模型导出的推理模型进行预测得到了相同的top1类别id，如下图：

![](https://ai-studio-static-online.cdn.bcebos.com/574ca8c71dbb48758ab750178228b25b45a47c9840df4c10905ca6b2539c8ea5)


```python
# 运行该指令即可启动模型推理
%cd /home/aistudio/PaddleVideo-develop/
!python3.7 tools/predict.py --input_file data/example_NTU-RGB-D_sketeton.npy \
                           --config configs/recognition/ctrgcn/ctrgcn_light_ntucs_joint.yaml \
                           --model_file inference/CTRGCN_light_joint_my/CTRGCN_joint.pdmodel \
                           --params_file inference/CTRGCN_light_joint_my/CTRGCN_joint.pdiparams \
                           --use_gpu=True \
                           --use_tensorrt=False
%cd /home/aistudio/
```

# 五、TIPC基础链条功能验证
<font size="3" color="black" face ="幼圆">参考自大佬[txyugood](https://aistudio.baidu.com/aistudio/projectdetail/4483896),
该部分依赖auto_log，需要进行安装，安装方式如下：


```python
%cd /home/aistudio/
!git clone https://gitee.com/Double_V/AutoLog
%cd AutoLog/
!pip3 install -r requirements.txt
!python3 setup.py bdist_wheel
!pip3 install ./dist/auto_log-1.2.0-py3-none-any.whl
```

<font size="3" color="black" face ="幼圆">对PaddleVideo-develop/test_tipc/configs/CTRGCN_light_joint/train_infer_python.txt进行修改以完成TIPC基础链条功能验证。
 修改后测试指令如下：


```python
%cd /home/aistudio/PaddleVideo-develop/
!bash test_tipc/prepare.sh test_tipc/configs/CTRGCN_light_joint/train_infer_python.txt 'lite_train_lite_infer'

!bash test_tipc/test_train_inference_python.sh test_tipc/configs/CTRGCN_light_joint/train_infer_python.txt 'lite_train_lite_infer'
```

<font size="3" color="black" face ="幼圆">测试结果如截图所示：
![](https://ai-studio-static-online.cdn.bcebos.com/c6a122f53e644e419ec7ccf6659d124c9c699c9ef416410985c270b577504532)
    
![](https://ai-studio-static-online.cdn.bcebos.com/47eb6d7cf341413d9fa80630c81a561de6977e3860554a538e36c273ce4b83aa)
    
也可以到PaddleVideo-develop/test_tipc/output/CTRGCN_light_joint/lite_train_lite_infer下查看相关日志。
    

# 六、总结
<font size="3" color="black" face ="幼圆">
1. 模型的轻量化不仅要考虑模型权重数量及浮点数运算量的减少，还要避免轻量化结构过于琐碎，同时还要保证足够的表达能力；
2. tipc基础链条功能验证的实现有点难，对比stgcn的实现和相关文档，勉强完成CTRGCN_light的验证。
