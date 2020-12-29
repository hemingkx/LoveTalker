import os
import json
import config
import pandas as pd
import numpy as np


def parse_raw_data():
    """返回情话的句子列表"""
    # 如果没有json文件，则从csv文件中生成json文件
    if not os.path.exists(config.data_path):
        data = pd.read_csv(config.origin_data_path, header=0)
        sents = data["微博正文"]
        sents = sents.tolist()
        # 按行写文件
        for sent in sents:
            with open(config.data_path, 'a+', encoding='utf-8') as f:
                line = json.dumps(sent, ensure_ascii=False)
                f.write(line + '\n')
    # 由于文件中有多行，直接读取会出现错误，因此一行一行读取
    file = open(config.data_path, 'r', encoding='utf-8')
    sents = []
    for sent in file.readlines():
        dic = json.loads(sent)
        sents.append(dic)
    return sents


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def get_data():
    """
    @return word2ix: dict,每个字对应的序号，形如u'月'->100
    @return ix2word: dict,每个序号对应的字，形如'100'->u'月'
    @return data: numpy数组，每一行是一句情话对应的字的下标
    """
    if os.path.exists(config.pickle_path):
        data = np.load(config.pickle_path, allow_pickle=True)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word

    # 如果没有处理好的二进制文件，则处理原始的json文件
    data = parse_raw_data()
    # 每一个字
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word: _ix for _ix, _word in enumerate(words)}
    # prepare for charRNN
    word2ix['<EOP>'] = len(word2ix)  # 终止标识符
    word2ix['<START>'] = len(word2ix)  # 起始标识符
    word2ix['</s>'] = len(word2ix)  # 空格
    ix2word = {_ix: _word for _word, _ix in list(word2ix.items())}

    # 为每句情话加上起始符和终止符
    for i in range(len(data)):
        data[i] = ["<START>"] + list(data[i]) + ["<EOP>"]

    # 将每句情话保存的内容由‘字’变成‘数’
    # 形如[春,江,花,月,夜]变成[1,2,3,4,5]
    new_data = [[word2ix[_word] for _word in _sentence]
                for _sentence in data]

    # 情话长度不够opt.maxlen的在前面补空格，超过的，删除末尾的
    pad_data = pad_sequences(new_data,
                             maxlen=config.maxlen,
                             padding='pre',
                             truncating='post',
                             value=len(word2ix) - 1)

    # 保存成二进制文件
    np.savez_compressed(config.pickle_path,
                        data=pad_data,
                        word2ix=word2ix,
                        ix2word=ix2word)
    return pad_data, word2ix, ix2word


if __name__ == "__main__":
    pad_data, word2ix, ix2word = get_data()
    print(word2ix)
