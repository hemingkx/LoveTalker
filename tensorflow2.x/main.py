import os
import config

from model import get_model
from data import for_fit, get_data

import numpy as np
import tensorflow as tf

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


class Evaluate(tf.keras.callbacks.Callback):
    """Save model of all epochs"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch训练完成后调用
        path = './checkpoint/model_{}.h5'.format(epoch)
        self.model.save(path)


def train():
    pad_data, word2ix, ix2word = get_data()
    step = len(pad_data) // config.batch_size  # 迭代步长
    model = get_model()
    model.fit_generator(for_fit(pad_data, True),
                        workers=1,
                        steps_per_epoch=step,
                        epochs=config.epoch,
                        callbacks=[Evaluate(model)])


def gen(model, start=''):
    """
    generate love words
    :param model: model to generate love words
    :param start: beginning of love words
    :return: love words
    """
    data, word2id, id2word = get_data()
    start = ["<START>"] + list(start)
    # word2id
    token_ids = [word2id[w] for w in start]

    while len(token_ids) < config.max_gen_len:
        output = model(np.array([token_ids, ], dtype=np.int32))
        prob = output.numpy()[0, -1, :]
        del output
        # 按照出现概率，对所有token倒序排列，取高概率的前一百个字
        p_args = prob.argsort()[::-1][:100]
        # 排列后的概率顺序
        p = prob[p_args]
        # 先对概率归一
        p = p / sum(p)
        # 再按照预测出的概率，随机选择一个词作为预测结果
        target_index = np.random.choice(len(p), p=p)
        target = p_args[target_index]
        # 保存
        token_ids.append(target)
        if target == word2id['<EOP>']:
            break
    return [id2word[w] for w in token_ids]


if __name__ == '__main__':
    train()
