import tensorflow as tf
from tensorflow.keras import layers, models

from xtructure.utils import rmsd_sqr


class Model(models.Model):
    def __init__(self, config):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(config['learning-rate'])
        self.vocab_size = config['vocab-size']
        self.embedding_size = config['embedding-size']
        network_config = config['network']
        self.kernel_size = network_config.get('kernel-size', 5)
        self.cnn_size = network_config.get('cnn-size', 256)
        self.cnn_num_layers = network_config.get('cnn-num-layers', 2)
        self.rnn_size = network_config.get('rnn-size', 512)
        self.hidden_size = network_config.get('hidden-size', 512)

        self.cnns = [
            layers.Conv1D(self.cnn_size, self.kernel_size, activation='relu')
            for _ in range(self.cnn_num_layers)
        ]
        self.flatten = layers.Flatten()
        self.iam_dense = layers.Dense(self.rnn_size, activation='relu')

        self.embedding = layers.Embedding(self.vocab_size, self.embedding_size)
        self.gru = layers.GRU(self.rnn_size, return_sequences=True)
        self.encoder_dense = layers.Dense(self.hidden_size, activation='relu')

        self.final_dense = layers.Dense(3)

    def call(self, input_iam, input_atomic_numbers, _bonds, training=False):
        # batch_size x iam_length x 1
        x = input_iam[:, :, None]
        for cnn in self.cnns:
            x = cnn(x)
        # batch_size x (cnn_size*n)
        x = self.flatten(x)
        # batch_size x rnn_size
        state = self.iam_dense(x)
        # batch_size x num_atoms x embedding_size
        embedded = self.embedding(input_atomic_numbers)
        # batch_size x num_atoms x rnn_size
        rnn_output = self.gru(embedded, initial_state=state)

        # batch_size x num_atoms x 3
        preds = self.final_dense(rnn_output)
        return preds

    def loss(self, preds, coords):
        return tf.reduce_mean(rmsd_sqr(preds, coords))

