# LoveTalker

本项目是基于charRNN的情话生成工具，适合新手入门。代码基于 [《深度学习框架PyTorch：入门与实践》](https://github.com/chenyuntc/pytorch-book) 编写。

## Function

#### 1. 基于前缀生成情话：

```
前缀：爱一个人
output：爱一个人，就是在一起很开心；爱一个人，就是一辈子
```

#### 2. 基于意境和前缀生成情话：

```
前缀：我想
意境：我的胸口有点闷，因为你堵在我心头了。
output：我想你一定很忙，所以你只看前三个字就好。
```

## Dataset

我们使用 [weibo-search](https://github.com/dataabc/weibo-search) 对微博话题 #情话# 下的数据进行爬取，具体爬取了2000-2020年共154380条文本数据。这些数据在love_word/data.csv文件中。这些数据只是初步爬取，是有很多噪声的。

## Requirements

代码基于pytorch实现，主要package配置包括：

- tqdm
- pytorch >= 1.5.1

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

