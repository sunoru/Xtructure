import time
import argparse

import tensorflow as tf
from tensorflow.python.keras.engine import training

from xtructure.init import get_dataset, init_model, load_config
from xtructure.utils import rmsd_sqr
from xtructure.visualization.main import visualize


def train_one_epoch(model, dataset, bonds, config, pre_train=False):
    num_inputs = len(dataset)
    batch_size = config['batch-size']
    logging_period = config['logging-period']
    bonds_loss = model.bonds_loss
    if pre_train:
        model.bonds_loss = None
        iam, atomic_numbers, coordinates = next(iter(dataset.batch(1)))
        training_data = tf.data.Dataset.from_tensor_slices(
            (iam, atomic_numbers, coordinates)
        ).repeat(20000).batch(batch_size)
    else:
        training_data = dataset.shuffle(num_inputs).batch(batch_size)
         
    for i, (input_iam, input_atomic_numbers, output_coordinates) in enumerate(training_data):
        with tf.GradientTape() as tape:
            preds = model(input_iam, input_atomic_numbers, bonds, training=not pre_train)
            loss = model.loss(preds, output_coordinates)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = loss.numpy()
        if i % logging_period == 0:
            print(f'Step {i:5d}: Loss = {loss}')
    if pre_train:
        model.bonds_loss = bonds_loss
    return loss


def train(model, config):
    print('Loading dataset...')
    bonds, dataset = get_dataset(config)
    print('Training...')
    start_time = time.perf_counter()
    if config['pre-train']:
        print('Pre-training...')
        train_one_epoch(model, dataset, bonds, config, pre_train=True)
        model.save_weights(config['checkpoints'] + '/pre-train')
    for i in range(config['epochs']):
        print(f'Epoch {i}:')
        loss = train_one_epoch(model, dataset, bonds, config)
        print(f'Train Loss = {loss}')
        model.save_weights(config['checkpoints'] + f'/{i}')
        print('Weights saved.')
    print('Training complete.')
    seconds = int(time.perf_counter() - start_time)
    print('Training Time:', f'{seconds // 60:d}:{seconds % 60:02d}')


def test(model, config):
    print('Loading test dataset...')
    bonds, dataset = get_dataset(config, False)
    print('Testing...')
    total_loss = 0
    batch_size = config['batch-size']
    for i, (input_iam, input_atomic_numbers, output_coordinates) in enumerate(
        dataset.batch(batch_size)
    ):
        preds = model(input_iam, input_atomic_numbers, bonds, training=False)
        loss = model.loss(preds, output_coordinates)
        total_loss += loss.numpy() * len(output_coordinates)
    print(f'Test Loss = {total_loss / len(dataset)}')

    return total_loss


def main(argv):
    parser = argparse.ArgumentParser(description='Xtructure')
    parser.add_argument('action', choices=['run', 'train', 'test', 'visualize'])
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--visualize-dir', type=str, help='visualization output directory', default='./visualization')
    args = parser.parse_args(argv[1:])
    config = load_config(args.config)
    if args.action == 'visualize':
        return visualize(config, args)
    train_only = args.action == 'train'
    test_only = args.action == 'test'
    model = init_model(config, test_only)
    if not test_only and config['has_train']:
        train(model, config)
    if not train_only and config['has_test']:
        test(model, config)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
