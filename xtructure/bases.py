import tensorflow as tf
from tensorflow.keras import models

from xtructure.utils import loss_bonds_exp, loss_bonds_sqr, loss_rmsd


class BaseModel(models.Model):
    def __init__(self, config):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(config['learning-rate'])
        self.bonds = None
        self.bonds_loss = config.get('bonds-loss', None)

    def set_bonds(self, bonds):
        if self.bonds is None:
            self.bonds = tf.constant(bonds)

    def loss(self, preds, coords):
        loss = loss_rmsd(preds, coords)
        if self.bonds_loss == 'sqr':
            loss += loss_bonds_sqr(preds, coords, self.bonds)
        elif self.bonds_loss == 'exp':
            loss += loss_bonds_exp(preds, coords, self.bonds)
        return loss
