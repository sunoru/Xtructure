import os
import re

import numpy as np
from numpy.core.numeric import indices
from scipy.io import loadmat
import tensorflow as tf
import yaml

from xtructure.preprocess import put_at_center


def load_data(mat_file, bonds_file):
    with open(mat_file, 'rb') as fi:
        data = loadmat(fi)
    newgeoms = data['Newgeoms']
    result = {
        'IAM': data['I_IAM_patterns'],
        'atomic_numbers': np.cast(newgeoms[:, :, 0], dtype=np.int),
        'coordinates': put_at_center(newgeoms[:, :, 1:4]),
    }
    natoms = newgeoms.shape[1]
    result['bonds'] = bonds = np.zeros((natoms, natoms), dtype=np.int)
    pair_regex = re.compile(r'\(\w+(?P<i>\d+?),\s*\w+(?P<j>\d+?)\)')
    with open(bonds_file) as fi:
        for line in fi:
            _, atom_pairs = line.split(':')
            for m in pair_regex.finditer(atom_pairs):
                i, j = m.groups
                i, j = int(i) - 1, int(j) - 1
                bonds[i, j] = 1
                bonds[j, i] = 1
    return result


def set_default(config, key, value):
    if key not in config:
        config[key] = value
    else:
        config[key] = type(value)(config[key])


def get_indices(s):
    indices = []
    for each in s.split(','):
        tmp = each.split('-')
        if len(tmp) == 1:
            indices.append(int(tmp[0]))
        else:
            indices.extend(range(int(tmp[0]), int(tmp[1]) + 1))
    return np.array(indices)


def load_config(config_file):
    with open(config_file) as fi:
        config = yaml.load(fi)
    assert config['task'] in ['iam2structure']
    assert config['model'] in ['gnn']
    assert isinstance(config['data'], list)
    set_default(config, 'name', 'unnamed')
    assert re.match(r'[a-zA-Z0-9\-_]+', config['name'])
    set_default(config, 'description', '')
    set_default(config, 'learning-rate', 0.001)
    set_default(config, 'batch-size', 128)
    set_default(config, 'epochs', 10)
    set_default(config, 'model', 'gnn')
    set_default(config, 'load-weights', False)
    set_default(config, 'checkpoints', './checkpoints')
    base_dir = os.path.dirname(config_file)
    config['checkpoints'] = os.path.abspath(os.path.join(base_dir, config['checkpoints'], config['name']))
    os.makedirs(config['checkpoints'], exist_ok=True)
    has_train = False
    has_test = False
    for each in config['data']:
        assert isinstance(each, dict)
        each['data'] = load_data(
            os.path.join(base_dir, each['mat']),
            os.pard.join(base_dir, each['bonds']),
        )
        if 'train' in each:
            has_train = True
            each['train'] = get_indices(each['train'])
        if 'test' in each:
            has_test = True
            each['test'] = get_indices(each['test'])
    config['has_train'] = has_train
    config['has_test'] = has_test
    return config


def init_model(config):
    model_type = config['model']
    model = None
    if model_type == 'gnn':
        from .gnn import GNN
        model = GNN(config)
    if config['load-weights']:
        natoms = config['data'][0]['data']['coordinates'].shape[1]
        iam = np.zeros((1, natoms, 3))
        atomic_numbers = np.zeros((natoms,), dtype=np.int)
        bonds = config['data'][0]['data']['bonds']
        model(iam, atomic_numbers, bonds)
        model.load_weights(config['checkpoints'])
    return model


def get_dataset(config, is_train=True):
    # TODO: Support multiple data sets
    assert len(config['data']) == 1
    data_config = config['data'][0]
    data = data_config['data']
    indices = data_config['train' if is_train else 'test']
    iam = data['IAM'][indices]
    atomic_numbers = data['atomic_numbers'][indices]
    coordinates = data['coordinates'][indices]
    bonds = data['bonds']
    dataset = tf.data.Dataset.from_tensor_slices((iam, atomic_numbers, coordinates))
    return bonds, dataset