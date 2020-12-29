# LoveTalker

本项目是基于charRNN的情话生成工具，适合新手入门。

- pytorch代码位于./pytorch目录下，基于 [《深度学习框架PyTorch：入门与实践》](https://github.com/chenyuntc/pytorch-book) 编写。
- tensorflow 2.x代码位于./tensorflow2.x目录下。

## Function

#### 1. 给定开头生成情话：

```
开头：爱一个人
output：爱一个人，就是在一起很开心；爱一个人，就是一辈子。
```

#### 2. 基于开头和语境生成情话：

语境不是情话的组成部分，但是为生成的情话提供了语气、句法格式等参考。

```
开头：我想
语境：我的胸口有点闷，因为你堵在我心头了。
output：我想你一定很忙，所以你只看前三个字就好。
```

## Dataset

我们使用 [weibo-search](https://github.com/dataabc/weibo-search) 对微博话题 #情话# 下的数据进行爬取，具体爬取了2000-2020年共154380条文本数据。这些数据在love_word/data.csv文件中。这些数据只是初步爬取，是有很多噪声的。

## Requirements

#### 1.pytorch

pytorch实现的主要package配置包括：

- tqdm
- pandas
- pytorch >= 1.5.1

安装环境可以运行：

```
pip install -r requirements.txt
```

#### 2.tensorflow

tensorflow2.x实现的主要package配置包括：

- pandas
- tensorflow-gpu >= 2.2.0

安装环境可以运行：

```
pip install -r requirements.txt
```

## Parameter Setting

模型参数等在config.py中进行设置。

## Usage

命令行输入

```
python main.py
```

模型运行结束后，每个epoch的模型保存在./checkpoints下。

