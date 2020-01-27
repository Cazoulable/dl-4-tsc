

from classifiers import classifiers

classifier_map = {
    'fcn': 'FCN',
    'mlp': 'MLP',
    'resnet': 'ResNet',
    # 'mcnn': 'MCNN',
    # 'tlenet': 'TLeNet',
    # 'twiesn': 'Twiesn',
    # 'encoder': 'Encoder',
    # 'mcdcnn': 'MCDCNN',
    # 'cnn': 'CNN',
    # 'inception': 'Inception'
}


def get_classifier(classifier_name):

    classifier_obj = getattr(classifiers, classifier_map[classifier_name])
    return classifier_obj


# def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
#     if classifier_name == 'fcn':
#         from classifiers import fcn
#         return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'mlp':
#         from classifiers import mlp
#         return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'resnet':
#         from classifiers import resnet
#         return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'mcnn':
#         from classifiers import mcnn
#         return mcnn.Classifier_MCNN(output_directory, verbose)
#     if classifier_name == 'tlenet':
#         from classifiers import tlenet
#         return tlenet.Classifier_TLENET(output_directory, verbose)
#     if classifier_name == 'twiesn':
#         from classifiers import twiesn
#         return twiesn.Classifier_TWIESN(output_directory, verbose)
#     if classifier_name == 'encoder':
#         from classifiers import encoder
#         return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'mcdcnn':
#         from classifiers import mcdcnn
#         return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'cnn':  # Time-CNN
#         from classifiers import cnn
#         return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
#     if classifier_name == 'inception':
#         from classifiers import inception
#         return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
