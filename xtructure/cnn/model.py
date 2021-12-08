import tensorflow as tf
from tensorflow.keras import layers, models
from xtructure.bases import BaseModel

from xtructure.utils import rmsd_sqr, rmsd_sqr_bonds, rmsd_exp_bonds


class Model(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        network_config = config['network']

        self.cnns = [
            layers.Dropout(x)
            if type(x) == float
            else layers.Conv1D(x[0], x[1], activation='relu')
            for x in network_config['cnns']
        ]
        self.flatten = layers.Flatten()
        self.denses = [
            layers.Dense(size, activation='relu')
            for size in network_config['denses']
        ]
        self.num_atoms = network_config['num-atoms']
        self.final_dense = layers.Dense(3 * self.num_atoms)
        self.bonds = None

    def call(self, input_iam, _input_atomic_numbers, bonds, training=False):
        self.set_bonds(bonds)
        # batch_size x iam_length x 1
        x = input_iam[:, :, None]
        for cnn in self.cnns:
            x = cnn(x, training=training)
        # batch_size x (cnn_size*n)
        x = self.flatten(x)
        # batch_size x rnn_size
        for dense in self.denses:
            x = dense(x)
        preds = self.final_dense(x)
        preds = tf.reshape(preds, [-1, self.num_atoms, 3])
        return preds - tf.reduce_mean(preds, axis=1)[:, None, :]

        # return tf.reduce_mean(rmsd_sqr(preds, coords))
