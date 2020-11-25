class Config(object):
    data_path = './love_word/data.json'  # 诗歌的文本文件存放路径
    pickle_path = 'love_word.npz'  # 预处理好的二进制文件
    constrain = None  # 长度限制
    lr = 1e-3
    weight_decay = 1e-4
    use_gpu = True
    epoch = 50
    batch_size = 128
    embedding_dim = 256
    hidden_dim = 512

    maxlen = 60  # 超过这个长度的之后字被丢弃，小于这个长度的在前面补空格
    max_gen_len = 80  # 生成情话最长长度
    debug_file = '/tmp/debug'
    model_path = 'checkpoints/love_word_29.pth'  # 预训练模型路径 'checkpoints/love_word_19.pth'
    prefix_words = None  # 不是诗歌的组成部分，用来控制生成诗歌的意境，如'我爱你'
    start_words = '希望'  # 诗歌开始
    model_prefix = 'checkpoints/love_word'  # 模型保存路径
