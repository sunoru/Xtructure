import tensorflow as tf

from xtructure.init import get_dataset, init_model, load_config

def train_one_epoch(model, dataset, bonds, config):
    total_loss = 0
    num_inputs = len(dataset)
    batch_size = config['batch-size']
    logging_period = config['logging-period']
    for i, (input_iam, input_atomic_numbers, output_coordinates) in enumerate(
        dataset.shuffle(num_inputs).batch(batch_size)
    ):
        with tf.GradientTape() as tape:
            predicted = model(input_iam, input_atomic_numbers, bonds, True)
            loss = model.loss(predicted, output_coordinates)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        loss = loss.numpy()
        if i % logging_period == 0:
            print(f"Step {i:5d}: Loss = {loss}")
        total_loss += loss
    return total_loss


def train(model, config):
    bonds, dataset = get_dataset(config)
    for i in range(config['epochs']):
        print(f'Epoch {i}:')
        loss = train_one_epoch(model, dataset, bonds, config)
        print(f'Total Loss = {loss}')
    print('Training complete.')
    model.save_weights(config['checkpoints'])
    print('Weights saved.')


def test(model, config):
    # TODO
    pass


def main(argv):
    if len(argv) != 1:
        print('Usage: xtructure config.yaml')
        return 1
    config = load_config(argv[0])
    model = init_model(config)
    if config['has_train']:
        train(model, config)
    else:
        test(model, config)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
