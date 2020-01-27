

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


def get_dataset(dataset_name):

    dataset_obj = getattr(datasets, dataset_map[dataset_name])
    return dataset_obj
