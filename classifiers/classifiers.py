# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers

import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
import time

from utils import utils

import matplotlib
matplotlib.use('agg')


class NetworkBasedClassifier:
    def __init__(self, output_directory, params):
        self.output_directory = output_directory
        self.input_shape = params['input_shape']
        self.nb_classes = params['nb_classes']
        self.verbose = params['verbose']

        # Learning rate parameters
        self.factor = 0.5
        self.patience = 50
        self.min_lr = 0.0001

        # Training parameters
        self.batch_size = 64
        self.nb_epochs = 1500

        self.callbacks = None
        self.model = None

    def initialize(self):
        # Define model
        self.set_callbacks()
        self.set_model()

        if self.verbose:
            self.model.summary()

        init_weight_file = self.get_file_path('model_init.hdf5')
        self.model.save_weights(init_weight_file)

    def get_file_path(self, file_name):
        return os.path.join(self.output_directory, file_name)

    def set_callbacks(self):
        model_file_path = self.get_file_path('best_model.hdf5')
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_file_path, monitor='loss', save_best_only=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=self.factor, patience=self.patience, min_lr=self.min_lr)
        self.callbacks = [reduce_lr, model_checkpoint]

    def set_model(self):
        raise("Model is not defined for class '{}'".format(self.__class__.__name__))

    def fit(self, x_train, y_train, x_val, y_val, y_true):
        if not tf.test.is_gpu_available:
            print('error')
            exit()

        # x_val and y_val are only used to monitor the test loss and NOT for training
        mini_batch_size = int(min(x_train.shape[0] / 10, self.batch_size))

        # Train model
        start_time = time.time()
        hist = self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=self.nb_epochs,
                              verbose=self.verbose, validation_data=(x_val, y_val), callbacks=self.callbacks)
        duration = time.time() - start_time

        # Save model
        model_file = self.get_file_path('last_model.hdf5')
        self.model.save(model_file)

        # Run prediction ...
        y_pred = self.predict(x_val, y_true, x_train, y_train, y_val, return_df_metrics=False)
        # ... and save them
        pred_file = self.get_file_path('y_pred.npy')
        np.save(pred_file, y_pred)

        # convert the predicted from binary to integer
        y_pred = np.argmax(y_pred, axis=1)

        # Save logs
        df_metrics = utils.save_logs(self.output_directory, hist, y_pred, y_true, duration)

        keras.backend.clear_session()
        return df_metrics

    def predict(self, x_test, y_true, x_train, y_train, y_test, return_df_metrics=True):
        best_model_path = self.get_file_path('best_model.hdf5')
        model = keras.models.load_model(best_model_path)

        start_time = time.time()
        y_pred = model.predict(x_test)
        if return_df_metrics:
            y_pred = np.argmax(y_pred, axis=1)
            df_metrics = utils.calculate_metrics(y_true, y_pred, 0.0)
            return df_metrics
        else:
            test_duration = time.time() - start_time
            duration_file = self.get_file_path('test_duration.csv')
            utils.save_test_duration(duration_file, test_duration)
            return y_pred


class ResNet(NetworkBasedClassifier):

    def __init__(self, output_directory, params):
        super(ResNet, self).__init__(output_directory, params)
        # Learning rate parameters
        self.factor = 0.5
        self.patience = 50
        self.min_lr = 0.0001

        # Training parameters
        self.batch_size = 64
        self.nb_epochs = 1500

    def set_model(self):
        n_feature_maps = 64
        input_layer = keras.layers.Input(self.input_shape)

        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


class FCN(NetworkBasedClassifier):
    def __init__(self, output_directory, params):
        super(FCN, self).__init__(output_directory, params)
        # # Learning rate parameters
        # self.factor = 0.5
        # self.patience = 50
        # self.min_lr = 0.0001

        # Training parameters
        self.batch_size = 16
        self.nb_epochs = 2000

    def set_model(self):
        input_layer = keras.layers.Input(self.input_shape)

        conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation(activation='relu')(conv1)

        conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        gap_layer = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(gap_layer)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])


class MLP(NetworkBasedClassifier):

    def __init__(self, output_directory, params):
        super(MLP, self).__init__(output_directory, params)
        # Learning rate parameters
        self.factor = 0.5
        self.patience = 200
        self.min_lr = 0.1

        # Training parameters
        self.batch_size = 16
        self.nb_epochs = 5000

    def set_model(self):
        input_layer = keras.layers.Input(self.input_shape)

        # flatten/reshape because when multivariate all should be on the same axis
        input_layer_flattened = keras.layers.Flatten()(input_layer)

        layer_1 = keras.layers.Dropout(0.1)(input_layer_flattened)
        layer_1 = keras.layers.Dense(500, activation='relu')(layer_1)

        layer_2 = keras.layers.Dropout(0.2)(layer_1)
        layer_2 = keras.layers.Dense(500, activation='relu')(layer_2)

        layer_3 = keras.layers.Dropout(0.2)(layer_2)
        layer_3 = keras.layers.Dense(500, activation='relu')(layer_3)

        output_layer = keras.layers.Dropout(0.3)(layer_3)
        output_layer = keras.layers.Dense(self.nb_classes, activation='softmax')(output_layer)

        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)
        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
