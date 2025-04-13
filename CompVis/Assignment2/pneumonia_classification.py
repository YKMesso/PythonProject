from __future__ import print_function

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.utils import class_weight
from keras.layers import RandomFlip, RandomRotation, RandomZoom
import keras_tuner as kt
from sklearn.metrics import classification_report

batch_size = 12
num_classes = 3
epochs = 3
trials = 3
img_width = 128
img_height = 128
img_channels = 3
fit = True

train_dir = 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\chest_xray\\train'
test_dir = 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\chest_xray\\test'

with tf.device('/gpu:0'):
    train_ds, val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    num_classes = len(class_names)

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])

    y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights_dict = {int(k): round(float(v), 2) for k, v in enumerate(class_weights)}

    def build_model(hp):
        model = Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(img_height, img_width, img_channels)))
        model.add(data_augmentation)
        model.add(Rescaling(1.0 / 255))

        for i in range(hp.Int('conv_layers', 1, 3)):
            model.add(Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=16, max_value=64, step=16),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
                activation='relu'
            ))
            model.add(MaxPooling2D())

        model.add(Flatten())
        model.add(Dense(
            units=hp.Int('dense_units', min_value=64, max_value=512, step=64),
            activation='relu'
        ))
        model.add(Dropout(rate=hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    tuner = kt.Hyperband(
        build_model,
        objective='val_accuracy',
        max_epochs=1,
        factor=5,
        directory='keras_tuner',
        project_name='pneumonia_classification'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    if fit:
        tuner.search(train_ds, validation_data=val_ds, epochs=1, callbacks=[stop_early])
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        model = tuner.hypermodel.build(best_hps)
        history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, class_weight=class_weights_dict)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    # SPEED-UP: Cut training epochs
    history = model.fit(
        train_ds.take(20),               # use a subset of data
        validation_data=val_ds.take(10),
        epochs=3,                        # reduce to 3 epochs
        class_weight=class_weights_dict
    )