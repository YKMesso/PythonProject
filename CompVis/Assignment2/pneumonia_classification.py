from __future__ import print_function

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np

import os
from collections import Counter

from keras.src.layers import RandomRotation
from sklearn.utils import class_weight
from keras.layers import RandomFlip, RandomRotation, RandomZoom
import keras_tuner as kt

# from 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\mnist_classification' import y_train


batch_size = 12
num_classes = 3
epochs = 8
img_width = 128
img_height = 128
img_channels = 3
fit = True  # make fit false if you do not want to train the network again

train_dir = 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\chest_xray\\train'
test_dir = 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\chest_xray\\test'


# Check if balanced
def count_images(dir):
    class_counts = {}
    for class_name in os.listdir(dir):
        class_path = os.path.join(dir, class_name)
        if os.path.isdir(class_path):
            count = len([file for file in os.listdir(class_path) if file.lower().endswith('.jpeg')])
            class_counts[class_name] = count
    return class_counts


train_count = count_images(train_dir)
test_count = count_images(test_dir)

print("Training Set Distribution:", train_count)
print("Test Set Distribution:", test_count)

#

with tf.device('/gpu:0'):
    # create training,validation and test datatsets
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
    print('Class Names: ', class_names)
    num_classes = len(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.show()

    data_augmentation = tf.keras.Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1)
    ])

    for images, _ in train_ds.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            augmented_image = data_augmentation(images[i:i + 1])  # Apply augmentation to a single image
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_image[0].numpy().astype("uint8"))
            plt.axis("off")
        plt.suptitle("Augmented Training Images")
        plt.show()

    # create model
    model = tf.keras.models.Sequential([
        data_augmentation,
        Rescaling(1.0 / 255),
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, img_channels)),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),  # flatten multidimensional outputs into single dimension for input to dense fully connected layers
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    # earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras", save_freq='epoch', save_best_only=True)

    # Class weighting
    y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)

    class_weights_dict = {int(k): round(float(v), 2) for k, v in enumerate(class_weights)}
    print("Class Weights:", class_weights_dict)

    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback],
            epochs=epochs,
            class_weight=class_weights_dict)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    # if shuffle=True when creating the dataset, samples will be chosen randomly
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    if fit:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')

    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(), 0))  # perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()] + '\nPredicted:{} {:.2f}%'.format(
                class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.show()
