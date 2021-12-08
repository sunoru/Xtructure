import tensorflow as tf


# N x NAtoms x 3
def put_at_center(coordinates):
    coordinates -= coordinates.mean(axis=1)[:, None, :]
    return coordinates


def rmsd_sqr(preds, target):
    return tf.reduce_sum(tf.reduce_mean(tf.square(preds - target), axis=1), axis=1)


def rmsd(preds, target):
    return tf.sqrt(rmsd_sqr(preds, target))


def rmsd_sqr_bonds(preds, coords, bonds):
    batch_size = preds.shape[0]
    n = len(bonds)
    loss = tf.zeros(batch_size)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bonds[i, j] > 0:
                loss += tf.reduce_sum(
                    tf.square(
                        tf.square(preds[:, i] - preds[:, j]) - 
                        tf.square(coords[:, i] - coords[:, j])
                    ),
                    axis=1
                )
    return loss


def rmsd_exp_bonds(preds, coords, bonds):
    batch_size = preds.shape[0]
    n = len(bonds)
    loss = tf.zeros(batch_size)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if bonds[i, j] > 0:
                loss += tf.reduce_sum(
                    tf.exp(tf.abs(
                        tf.square(preds[:, i] - preds[:, j]) - 
                        tf.square(coords[:, i] - coords[:, j])
                    )),
                    axis=1
                )
    return loss


def loss_rmsd(preds, coords):
    return tf.reduce_mean(rmsd_sqr(preds, coords))


def loss_bonds_exp(preds, coords, bonds):
    return tf.reduce_mean(rmsd_exp_bonds(preds, coords, bonds))


def loss_bonds_sqr(preds, coords, bonds):
    return tf.reduce_mean(rmsd_sqr_bonds(preds, coords, bonds))
