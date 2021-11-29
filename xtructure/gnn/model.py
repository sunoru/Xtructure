import tensorflow as tf
from tensorflow.keras import layers, models

class Model(models.Model):
    def __init__(self, config):
        super().__init__()

        self.optimizer = tf.keras.optimizers.Adam(config['learning-rate'])

        self.embedding = layers.Embedding(config['vocab-size'], config['embedding-size']),


    def call(self, input_iam, input_atomic_numbers, bonds, training=False):
        pass

    def loss(self, predictions, labels):
        pass
