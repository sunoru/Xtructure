import os
from xtructure.init import get_dataset, get_indices, init_model
from xtructure.visualization.utils import write_xyz


def visualize(config, args, indices=None):
    model = init_model(config, True)
    output_dir = os.path.join(args.visualize_dir, config['name'])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f'Output: {output_dir}')
    print('Loading test dataset...')
    bonds, dataset = get_dataset(config, False)
    dataset = list(dataset)
    n = len(dataset)
    if indices is None:
        indices = get_indices(input(f'Input indices to visualize (0-{n-1}): '))
    for i in indices:
        print(f'Predicting structure for index {i}... ', end='')
        input_iam, input_atomic_numbers, output_coordinates = dataset[i]
        preds = model(input_iam[None, :], input_atomic_numbers[None, :], bonds, True)
        loss = model.loss(preds, output_coordinates[None, :])
        print(f'Loss = {loss}')
        write_xyz(os.path.join(output_dir, f'{i}-predicted.xyz'), input_atomic_numbers.numpy(), preds[0].numpy())
        write_xyz(os.path.join(output_dir, f'{i}-real.xyz'), input_atomic_numbers.numpy(), output_coordinates.numpy())

    print('Done.')
    return 0
