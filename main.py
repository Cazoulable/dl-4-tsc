
import argparse
import os

from classifiers import classifier_factory
from datasets import dataset_factory
from utils import utils

# change this directory for your machine
EXPERIMENTS_DIR = '/Users/simoncazals/experiments/parkinson'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_id', type=int, help='Experiment ID.')
    args = parser.parse_args()

    # Load configuration file
    experiment_dir = os.path.join(EXPERIMENTS_DIR, '{:04d}'.format(args.experiment_id))
    config_file = os.path.join(experiment_dir, 'config.yaml')
    config = utils.parse_config_file(config_file)

    if os.path.exists(os.path.join(experiment_dir, 'DONE')):
        print("Experiment #{:04d} has already train. Specify a different ID.".format(args.experiment_id))

    else:
        # Initialize dataset
        dataset_obj = dataset_factory.get_dataset(config['dataset']['name'])
        dataset = dataset_obj(config['dataset'])
        dataset.initialize()

        # Get data generator
        train_generator, val_generator = dataset.get_generators(**config['data'])
        input_shape = train_generator.input_shape
        n_classes = train_generator.n_classes

        # Get classifier
        classifier_obj = classifier_factory.get_classifier(config['classifier']['name'])
        classifier = classifier_obj(experiment_dir, input_shape=input_shape, n_classes=n_classes, verbose=True)
        classifier.initialize()

        # Train model
        classifier.fit(train_generator, val_generator, **config['training'])
        print('DONE')

        # the creation of this directory means
        utils.create_directory(os.path.join(experiment_dir, 'DONE'))
