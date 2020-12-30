import config
import tensorflow as tf


def get_model():
    model = tf.keras.Sequential([
        # 不定长度的输入
        tf.keras.layers.Input((None,)),
        # 词嵌入层
        tf.keras.layers.Embedding(input_dim=config.vocab_size, output_dim=config.embedding_dim),
        # 第一个LSTM层，返回序列作为下一层的输入
        tf.keras.layers.LSTM(config.hidden_dim, return_sequences=True),
        # 第二个LSTM层，返回序列作为下一层的输入
        tf.keras.layers.LSTM(config.hidden_dim, return_sequences=True),
        # 第三个LSTM层，返回序列作为下一层的输入
        tf.keras.layers.LSTM(config.hidden_dim, return_sequences=True),
        # 第四个LSTM层，返回序列作为下一层的输入
        tf.keras.layers.LSTM(config.hidden_dim, return_sequences=True),
        # 对每一个时间点的输出都做softmax，预测下一个词的概率
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(config.vocab_size, activation='softmax')),
    ])

    # 查看模型结构
    model.summary()

    # 配置优化器和损失函数
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=config.lr),
                  loss=tf.keras.losses.categorical_crossentropy)
    return model


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "1, 2, 3"
    model = get_model()
