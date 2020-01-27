
import numpy as np
import os
import sys
import sklearn

from classifiers import classifier_factory
from utils import constants
from utils import utils

# change this directory for your machine
ROOT_DIR = '/b/home/uha/hfawaz-datas/dl-tsc-temp/'


def fit_classifier(classifier_name, datasets, output_directory, verbose=False):
    x_train, y_train, x_test, y_test = datasets
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save original y because later we will use binary
    y_true = np.argmax(y_test, axis=1)

    if len(x_train.shape) == 2:  # if uni-variate
        # add a dimension to make it multivariate with one dimension 
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    input_shape = x_train.shape[1:]
    classifier_params = {
        'input_shape': input_shape,
        'nb_classes': nb_classes,
        'verbose': verbose
    }

    classifier_obj = classifier_factory.get_classifier(classifier_name)
    classifier = classifier_obj(output_directory, classifier_params)
    classifier.fit(x_train, y_train, x_test, y_test, y_true)


# main
if sys.argv[1] == 'run_all':
    for classifier_name in constants.CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in constants.ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)

            datasets_dict = utils.read_all_datasets(ROOT_DIR, archive_name)

            for _iter in range(constants.ITERATIONS):
                print('\t\titer : {}'.format(_iter))

                trr = ''
                if _iter != 0:
                    trr = '_itr_{}'.format(_iter)

                tmp_output_directory = os.path.join(ROOT_DIR, 'results', classifier_name, archive_name + trr)

                for dataset_name in constants.dataset_names_for_archive[archive_name]:
                    print('\t\t\tdataset_name: ', dataset_name)

                    output_directory = os.path.join(tmp_output_directory, dataset_name)
                    utils.create_directory(output_directory)

                    dataset = datasets_dict[dataset_name]
                    fit_classifier(classifier_name, dataset, output_directory)

                    print('\t\t\t\tDONE')

                    # the creation of this directory means
                    utils.create_directory(os.path.join(output_directory, 'DONE'))

elif sys.argv[1] == 'transform_mts_to_ucr_format':
    utils.transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    utils.visualize_filter(ROOT_DIR)
elif sys.argv[1] == 'viz_for_survey_paper':
    utils.viz_for_survey_paper(ROOT_DIR)
elif sys.argv[1] == 'viz_cam':
    utils.viz_cam(ROOT_DIR)
elif sys.argv[1] == 'generate_results_csv':
    res = utils.generate_results_csv('results.csv', ROOT_DIR)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]

    if itr == '_itr_0':
        itr = ''

    output_directory = os.path.join(ROOT_DIR, 'results', classifier_name, archive_name + itr, dataset_name)
    test_dir_df_metrics = os.path.join(output_directory, 'df_metrics.csv')

    print('Method: ', archive_name, dataset_name, classifier_name, itr)

    if os.path.exists(test_dir_df_metrics):
        print('Already done')
    else:
        utils.create_directory(output_directory)
        dataset = utils.read_dataset(ROOT_DIR, archive_name, dataset_name)

        fit_classifier(classifier_name, dataset, output_directory)
        print('DONE')

        # the creation of this directory means
        utils.create_directory(os.path.join(output_directory, 'DONE'))
