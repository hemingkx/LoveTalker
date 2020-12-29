import torch

# 文件位置
origin_data_path = './love_word/data.csv'  # 情话的文本文件存放路径
data_path = './love_word/data.json'  # 诗歌的json文件存放路径
pickle_path = './love_word/love_word.npz'  # 预处理好的二进制文件

# 学习率设置
lr = 1e-3
weight_decay = 1e-4
epoch = 50
lr_step = 10
lr_gamma = 0.8

# batch size
batch_size = 128

# 模型参数
embedding_dim = 256
hidden_dim = 512
num_layers = 4

# 情话生成设置
maxlen = 60  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
max_gen_len = 80  # 生成情话最长长度
model_path = 'checkpoints/love_word_0.pth'  # 预训练模型路径 'checkpoints/love_word_19.pth'
prefix_words = None  # 生成情话的语境，如'我爱你'
start_words = '希望'  # 情话的开头部分
model_prefix = 'checkpoints/love_word'  # 模型保存路径

gpu = '1'

# 设置gpu为命令行参数指定的id
if gpu != '':
    device = torch.device(f"cuda:{gpu}")
else:
    device = torch.device("cpu")
