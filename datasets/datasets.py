
import keras
import math
import numpy as np
import os
import pandas as pd
import random
from tensorflow.keras.utils import Sequence

DATA_DIR = '/Users/simoncazals/Desktop/data/parkinson/'


def capitalize(name):
    return name[0].upper() + name[1:]


class ParkinsonDataset:
    def __init__(self, args):
        assert args['study'] in ['CIS', 'REAL'], "'study_name' must be 'CIS' or 'REAL'"
        assert args['partition'] in ['training', 'ancillary'], "'split' must be 'training' or 'ancillary'"

        self.study_name = args['study']
        self.partition_name = args['partition']

        self.data = None
        self.labels = None
        self.measurement_ids = None
        self.measurement_split = None

    def initialize(self):
        print("Initialize dataset ...")
        data_dir = os.path.join(DATA_DIR, '{}-PD_{}_data'.format(self.study_name, self.partition_name))
        labels_file = os.path.join(DATA_DIR, '{}-PD_data_labels'.format(self.study_name),
                                   '{}-PD_{}_Data_IDs_Labels.csv'.format(self.study_name, capitalize(self.partition_name)))

        labels_df = pd.read_csv(labels_file, index_col='measurement_id')
        self.labels = labels_df.to_dict('index')

        # Load measurement
        data = {}
        for measurement_id in self.labels:
            measurement_file = os.path.join(data_dir, '{}.csv'.format(measurement_id))
            assert os.path.exists(measurement_file), "Measurement file '{}' does not exist".format(measurement_file)
            measurement_df = pd.read_csv(measurement_file)
            xs = list(measurement_df.X)
            ys = list(measurement_df.Y)
            zs = list(measurement_df.Z)
            data[measurement_id] = np.array([xs, ys, zs]).T
        self.data = data
        print("Done !")

    def get_generators(self, label_name, batch_size, sample_length, split_ratio=0.8):
        assert label_name in ['on_off', 'dyskinesia', 'tremor'], \
            "label name '{}' must be in 'on_off', 'dyskinesia' or 'tremor'".format(label_name)

        # Data split train/test
        self.measurement_ids = [m_id for m_id in self.labels if not math.isnan(self.labels[m_id][label_name])]
        n_measurements = len(self.measurement_ids)
        print("{} measurement used for training ".format(n_measurements))
        random.shuffle(self.measurement_ids)
        train_measurement_ids = self.measurement_ids[:int(n_measurements * split_ratio)]
        test_measurement_ids = self.measurement_ids[int(n_measurements * split_ratio):]

        # Create generators
        x_train = [self.data[measurement_id] for measurement_id in train_measurement_ids]
        y_train = [int(self.labels[measurement_id][label_name]) for measurement_id in train_measurement_ids]
        train_generator = ParkinsonDatasetGenerator(x_train, y_train, batch_size, sample_length)

        x_test = [self.data[measurement_id] for measurement_id in test_measurement_ids]
        y_test = [int(self.labels[measurement_id][label_name]) for measurement_id in test_measurement_ids]
        test_generator = ParkinsonDatasetGenerator(x_test, y_test, batch_size, sample_length)

        return train_generator, test_generator


class ParkinsonDatasetGenerator(Sequence):
    """Generates data for Keras"""

    def __init__(self, data, labels, batch_size, sample_length, shuffle=True):
        """Initialization"""
        # Data and labels
        self.data = data
        self.labels = labels
        self.n_measurements = len(self.labels)

        # Input and output dimensions
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.n_channels = 3
        self.n_classes = 5

        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.floor(self.n_measurements / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        x, y = self.__data_generation(indexes)

        return x, y

    @property
    def input_shape(self):
        return self.sample_length, self.n_channels

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(self.n_measurements)

        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        """Generates data containing batch_size samples"""
        x = np.empty((self.batch_size, self.sample_length, self.n_channels))
        y = np.empty(self.batch_size, dtype=int)

        # Generate data
        for i, idx in enumerate(indexes):
            # Store sample
            sample_idx = np.random.randint(self.data[idx].shape[0] - self.sample_length)
            x[i, ] = self.data[idx][sample_idx: sample_idx + self.sample_length]

            # Store class
            y[i] = self.labels[idx]

        return x, keras.utils.to_categorical(y, num_classes=self.n_classes)
