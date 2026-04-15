## 🗝️开始使用
### `一、安装`
此仓库的代码是在 Linux 系统下运行的。我们尚未测试是否能在其他操作系统下运行。

首先需要安装[VMama仓库](https://github.com/MzeroMiko/VMamba)。以下安装顺序取自VMama仓库。


**步骤 1 —— 克隆仓库:**

克隆该版本库并导航至项目目录：

**步骤 2 —— 环境设置:**

建议设置 conda 环境并通过 pip 安装依赖项。使用以下命令设置环境：

***创建并激活新的 conda 环境***


***安装依赖项***


***检测和分割任务的依赖库（在 VMamba 中为可选项）***

### `三、数据准备`
***二元变化检测***

论文使用了三个基准数据集 [SYSU](https://github.com/liumency/SYSU-CD)、[LEVIR-CD+](https://chenhao.in/LEVIR/) 和 [WHU-CD](http://gpcv.whu.edu.cn/data/building_dataset.html) 用于评估模型的二元变化检测的性能。请下载这些数据集，并将其组织成下述文件夹/文件结构：

```
${DATASET_ROOT}   # 数据集根目录，例如: /home/username/data/SYSU
├── train
│   ├── T1
│   │   ├──00001.png
│   │   ├──00002.png
│   │   ├──00003.png
│   │   ...
│   │
│   ├── T2
│   │   ├──00001.png
│   │   ... 
│   │
│   └── GT
│       ├──00001.png 
│       ...   
│   
├── val
│   ├── ...
│   ...
│
├── test
│   ├── ...
│   ...
│ 
├── train.txt   # 数据名称列表，记录所有训练数据的名称
├── val.txt     # 数据名称列表，记录所有验证数据的名称
└── test.txt    # 数据名称列表，记录所有测试数据的名称
```


### `四、训练模型`
在训练模型之前，请进入 [`changedetection`]文件夹，其中包含网络定义、训练和测试的所有代码。

