import time

import tensorflow as tf

from xtructure.init import get_dataset, init_model, load_config
from xtructure.utils import rmsd_sqr

def train_one_epoch(model, dataset, bonds, config):
    num_inputs = len(dataset)
    batch_size = config['batch-size']
    logging_period = config['logging-period']
    for i, (input_iam, input_atomic_numbers, output_coordinates) in enumerate(
        dataset.shuffle(num_inputs).batch(batch_size)
    ):
        with tf.GradientTape() as tape:
            preds = model(input_iam, input_atomic_numbers, bonds, training=True)
            loss = model.loss(preds, output_coordinates)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = loss.numpy()
        if i % logging_period == 0:
            print(f"Step {i:5d}: Loss = {loss}")
    return loss


def train(model, config):
    print("Loading dataset...")
    bonds, dataset = get_dataset(config)
    print("Training...")
    start_time = time.perf_counter()
    for i in range(config['epochs']):
        print(f'Epoch {i}:')
        loss = train_one_epoch(model, dataset, bonds, config)
        print(f'Train Loss = {loss}')
    print('Training complete.')
    seconds = int(time.perf_counter() - start_time)
    print("Training Time:", f"{seconds // 60:d}:{seconds % 60:02d}")
    model.save_weights(config['checkpoints'])
    print('Weights saved.')


def test(model, config):
    print("Loading test dataset...")
    bonds, dataset = get_dataset(config, False)
    print("Testing...")
    total_loss = 0
    batch_size = config['batch-size']
    for i, (input_iam, input_atomic_numbers, output_coordinates) in enumerate(
        dataset.batch(batch_size)
    ):
        preds = model(input_iam, input_atomic_numbers, bonds, True)
        loss = model.loss(preds, output_coordinates)
        total_loss += loss.numpy() * len(output_coordinates)
    print(f'Test Loss = {total_loss / len(dataset)}')
    print(preds[-2:])
    print(output_coordinates[-2:])
    print(rmsd_sqr(preds[-2:], output_coordinates[-2:]))

    return total_loss


def main(argv):
    argc = len(argv)
    if argc not in [2, 3] or argc > 2 and argv[2] != '--test':
        print('Usage: xtructure config.yaml [--test]')
        return 1
    config = load_config(argv[1])
    test_only = argc > 2
    model = init_model(config, test_only)
    if not test_only and config['has_train']:
        train(model, config)
    if config['has_test']:
        test(model, config)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
