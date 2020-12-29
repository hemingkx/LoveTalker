import sys
import config
from tqdm import tqdm
from data import get_data
from model import PoetryModel

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


def generate(model, start_words, ix2word, word2ix, prefix_words=None):
    """
    给定几个词，根据这几个词接着生成一句完整的情话
    start_words：u'今天我很开心'
    """

    results = list(start_words)
    start_word_len = len(start_words)
    # 手动设置第一个词为<START>
    input = torch.Tensor([word2ix['<START>']]).view(1, 1).long()
    input = input.to(config.device)
    hidden = None

    if prefix_words:
        for word in prefix_words:
            output, hidden = model(input, hidden)
            input = input.data.new([word2ix[word]]).view(1, 1)

    for i in range(config.max_gen_len):
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


def train():
    # 获取数据
    data, word2ix, ix2word = get_data()
    data = torch.from_numpy(data)
    dataloader = DataLoader(data, batch_size=config.batch_size, shuffle=True, num_workers=1)

    # 模型定义
    model = PoetryModel(len(word2ix), config.embedding_dim, config.hidden_dim, config.num_layers)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = StepLR(optimizer, step_size=config.lr_step, gamma=config.lr_gamma)
    criterion = nn.CrossEntropyLoss()
    model.to(config.device)

    for epoch in range(config.epoch):
        total_loss = 0
        for data_ in tqdm(dataloader):
            # 训练
            data_ = data_.long().transpose(1, 0).contiguous()
            data_ = data_.to(config.device)
            optimizer.zero_grad()
            input_, target = data_[:-1, :], data_[1:, :]
            output, _ = model(input_)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print("epoch: ", epoch, "loss: ", total_loss / len(dataloader))
        torch.save(model.state_dict(), '%s_%s.pth' % (config.model_prefix, epoch))


def gen():
    """生成相应的情话"""
    data, word2ix, ix2word = get_data()
    model = PoetryModel(len(word2ix), config.embedding_dim, config.hidden_dim, config.num_layers)
    state_dict = torch.load(config.model_path, map_location=lambda s, l: s)
    model.load_state_dict(state_dict)
    model.to(config.device)

    # python2和python3 字符串兼容
    if sys.version_info.major == 3:
        if config.start_words.isprintable():
            start_words = config.start_words
            prefix_words = config.prefix_words if config.prefix_words else None
        else:
            start_words = config.start_words.encode('ascii', 'surrogateescape').decode('utf8')
            prefix_words = config.prefix_words.encode('ascii', 'surrogateescape').decode(
                'utf8') if config.prefix_words else None
    else:
        start_words = config.start_words.decode('utf8')
        prefix_words = config.prefix_words.decode('utf8') if config.prefix_words else None

    start_words = start_words.replace(',', u'，').replace('.', u'。').replace('?', u'？')

    result = generate(model, start_words, ix2word, word2ix, prefix_words)
    print(''.join(result))


if __name__ == '__main__':
    train()
    gen()