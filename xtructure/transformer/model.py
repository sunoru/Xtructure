import tensorflow as tf
from tensorflow.keras import layers, models

from xtructure.transformer import attention
from xtructure.utils import rmsd_sqr


class Model(models.Model):
    def __init__(self, config):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(config['learning-rate'])
        self.vocab_size = config['vocab-size']
        self.embedding_size = config['embedding-size']
        # No decoder is needed.
        network_config = config['network']
        self.window_size = network_config['window-size']
        self.encoder_embedding = layers.Embedding(self.window_size, self.embedding_size)
        self.decoder_embedding = layers.Embedding(self.window_size, self.embedding_size)
        self.encoder_positional = attention.PositionEncoding(self.window_size, self.embedding_size)
        self.decoder_positional = attention.PositionEncoding(self.window_size, self.embedding_size)
        self.num_encoders = network_config['num-encoders']
        self.num_decoders = network_config['num-decoders']
        self.encoders = [
            attention.TransformerBlock(self.embedding_size, False)
            for _ in range(self.num_encoders)
        ]
        self.decoders = [
            attention.TransformerBlock(self.embedding_size, True)
            for _ in range(self.num_decoders)
        ]
        self.final_dense = layers.Dense(3)

    def call(self, input_iam, input_atomic_numbers, _bonds, training=False):
        encoder_embedded = self.encoder_embedding(input_atomic_numbers)
        encoder_output = self.encoder_positional(encoder_embedded, input_iam, training=training)
        for encoder in self.encoders:
            encoder_output = encoder(encoder_output, training=training)

        decoder_embedded = self.decoder_embedding(input_atomic_numbers)
        decoder_output = self.decoder_positional(decoder_embedded, input_iam)
        for decoder in self.decoders:
            decoder_output = decoder(decoder_output, context=encoder_output, training=training)
        preds = self.final_dense(decoder_output)
        return preds

    def loss(self, preds, coords):
        return tf.reduce_mean(rmsd_sqr(preds, coords))

