import os
import config

from model import get_model
from data import get_data

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# 设置GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# 获取所有GPU组成list
gpu = tf.config.experimental.list_physical_devices('GPU')
# 设置按需申请，由于我这里仅有一块GPU,multi-GPU需要for一下
tf.config.experimental.set_memory_growth(gpu[0], True)


class Evaluate(tf.keras.callbacks.Callback):
    """Save model of all epochs"""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        # 在每个epoch训练完成后调用
        path = './checkpoints/model_{}.h5'.format(epoch)
        self.model.save(path)


def one_hot(x, y):
    return x, tf.one_hot(y, config.vocab_size)


def train():
    pad_data, word2ix, ix2word = get_data()
    dataset = tf.data.Dataset.from_tensor_slices((pad_data[:, :-1], pad_data[:, 1:]))
    dataset_one_hot = dataset.map(one_hot)
    train_dataset = dataset_one_hot.batch(config.batch_size)
    model = get_model()
    model.fit(train_dataset,
              epochs=config.epoch,
              callbacks=[Evaluate(model)])


def gen(model_id, prefix='', start=''):
    """
    generate love words
    :param model_id: model of epoch model_id
    :param prefix: prefix of love words
    :param start: beginning of love words
    :return: love words
    """
    model = load_model('./checkpoints/model_{}.h5'.format(model_id))
    data, word2id, id2word = get_data()
    start = ["<START>"] + list(prefix) + list(start)
    # word2id
    token_ids = [word2id[w] for w in start]

    while len(token_ids) < config.max_gen_len:
        output = model(np.array([token_ids, ], dtype=np.int32))
        prob = output.numpy()[0, -1, :]
        del output
        # 按照出现概率，对所有token倒序排列，取最高概率的字
        target = prob.argsort()[::-1][0]
        # 保存
        if target == word2id['<EOP>']:
            break
        else:
            token_ids.append(target)
    result = [id2word[w] for w in token_ids[len(prefix) + 1:]]
    print(''.join(result))


if __name__ == '__main__':
    # train()
    gen(49, config.prefix_words, config.start_words)

