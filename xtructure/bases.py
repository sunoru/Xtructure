import tensorflow as tf
from tensorflow.keras import models

from xtructure.utils import loss_bonds_exp, loss_bonds_sqr, loss_rmsd


class BaseModel(models.Model):
    def __init__(self, config):
        super().__init__()
        self.optimizer = tf.keras.optimizers.Adam(config['learning-rate'])
        self.bonds = None
        bonds_loss = config.get('bonds-loss', None)
        if bonds_loss is None:
            self.bonds_loss = None
        else:
            t = bonds_loss.split('@')
            self.bonds_loss = t[0]
            self.bonds_scale = float(t[1]) if len(t) > 1 else 1.0

    def set_bonds(self, bonds):
        if self.bonds is None:
            self.bonds = tf.constant(bonds)

    def loss(self, preds, coords):
        loss = loss_rmsd(preds, coords)
        if self.bonds_loss == 'sqr':
            loss += self.bonds_scale * loss_bonds_sqr(preds, coords, self.bonds)
        elif self.bonds_loss == 'exp':
            loss += self.bonds_scale * loss_bonds_exp(preds, coords, self.bonds)
        return loss
