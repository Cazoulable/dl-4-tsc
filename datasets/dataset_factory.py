

from datasets import datasets

dataset_map = {
    'parkinson': 'ParkinsonDataset',
    # 'mlp': 'MLP',
    # 'resnet': 'ResNet',
    # 'mcnn': 'MCNN',
    # 'tlenet': 'TLeNet',
    # 'twiesn': 'Twiesn',
    # 'encoder': 'Encoder',
    # 'mcdcnn': 'MCDCNN',
    # 'cnn': 'CNN',
    # 'inception': 'Inception'
}


def get_dataset(dataset_config):

    dataset_obj = getattr(datasets, dataset_map[dataset_config['name']])
    return dataset_obj(dataset_config)

