import tensorflow as tf


# N x NAtoms x 3
def put_at_center(coordinates):
    coordinates -= coordinates.mean(axis=1)[:, None, :]
    return coordinates


def rmsd_sqr(preds, target):
    return tf.reduce_sum(tf.reduce_mean(tf.square(preds - target), axis=1), axis=1)


def rmsd(preds, target):
    return tf.sqrt(rmsd_sqr(preds, target))
