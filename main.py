# coding:utf8
import sys
import tqdm
from data import get_data
from model import PoetryModel
from config import Config
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

opt = Config()


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一句完整的情话
    start_words：u'今天我很开心'
    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    if opt.use_gpu:
        input = input.cuda()
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(opt.max_gen_len):
        output, hidden = model(input, hidden)

        if i < start_word_len:
            w = results[i]
            input = input.data.new([word2ix[w]]).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0].item()
            w = ix2word[top_index]
            results.append(w)
            input = input.data.new([top_index]).view(1, 1)
        if w == '<EOP>':
            del results[-1]
            break
    return results


def train(**kwargs):
    for k, v in kwargs.items():
        setattr(opt, k, v)

    opt.device = torch.device('cuda:3') if opt.use_gpu else torch.device('cpu')
    device = opt.device

    # 获取数据
    data, word2ix, ix2word = get_data(opt)
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    if opt.model_path:
        model.load_state_dict(torch.load(opt.model_path))
    model.to(device)

    for epoch in range(opt.epoch):
        total_loss = 0
        for ii, data_ in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("epoch: ", epoch, "loss: ", total_loss / len(dataloader))
        torch.save(model.state_dict(), '%s_%s.pth' % (opt.model_prefix, epoch))


def gen(**kwargs):
    """
    提供命令行接口，用以生成相应的情话
    """

    for k, v in kwargs.items():
        setattr(opt, k, v)
    data, word2ix, ix2word = get_data(opt)
    model = PoetryModel(len(word2ix), Config.embedding_dim, Config.hidden_dim)
    map_location = lambda s, l: s
    state_dict = torch.load(opt.model_path, map_location=map_location)
    model.load_state_dict(state_dict)

    if opt.use_gpu:
        model.cuda()

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if opt.start_words.isprintable():
            start_words = opt.start_words
            prefix_words = opt.prefix_words if opt.prefix_words else None
        else:
            start_words = opt.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = opt.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if opt.prefix_words else None
    else:
        start_words = opt.start_words.decode('utf8')
        prefix_words = opt.prefix_words.decode('utf8') if opt.prefix_words else None

    start_words = start_words.replace(',', u'，') \
        .replace('.', u'。') \
        .replace('?', u'？')

    result = generate(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))


if __name__ == '__main__':
    # train()
    gen()