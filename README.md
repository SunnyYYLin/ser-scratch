# Speech Emotion Recognition from Scratch

这是一个使用深度学习模型进行语音情感识别的项目。报告在`report/`下。

## 目录结构

```plaintext
.
├── LICENSE                # 许可证
├── README.md              # 项目说明文件
├── checkpoints/           # 模型检查点
├── config.py              # 模型配置文件
├── data_process.py        # 数据处理和数据集加载脚本
├── logs/                  # 日志文件
├── metrics.py             # 模型评估指标定义
├── models                 # 模型定义目录
│   ├── __init__.py        # 模型包初始化文件
│   ├── backbone.py        # 主干网络的选择
│   ├── cnn.py             # 卷积神经网络主干
│   ├── emotion_classifier.py # 情感分类器定义
│   └── wav2vec.py         # wav2vec2主干
├── train.py               # 训练模型
├── ui                     # Web界面相关代码目录
│   ├── __init__.py        
│   ├── events.py          # 事件处理
│   └── ui.py              # UI界面定义
└── webui.py               # 启动Web界面
```

## 安装和配置

1. 克隆此仓库：
```sh
git clone https://github.com/SunnyYYLin/ser-scratch.git
cd ser-scratch
```

2. 创建并激活虚拟环境：
```sh
conda create -n ser python=3.11
conda activate ser
```

3. 安装依赖：
```sh
pip install -r requirements.txt
```
或者是
```sh
pip install torch torchaudio transformers[torch] datasets torchmetrics tensorboard librosa soundfile gradio
```

## 数据集与预训练模型下载

请确保可以连接到Huggingface或者其镜像网站，数据集和预训练模型会自动下载。

## 使用

1. 训练模型：指定`train.py`中的`config`，运行`python train.py`即可训练对应模型。模型日志记录在`logs/`，权重保存在`checkpoints/`。

2. 启动SER图形界面系统：运行`python webui.py`即可。