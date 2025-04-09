import tensorflow as tf
import numpy as np
from sklearn.utils import class_weight

# Update this to your actual train directory
train_dir = 'C:\\Users\\messo\\PycharmProjects\\PythonProject\\CompVis\\Assignment2\\chest_xray\\train'


batch_size = 12
img_height = 128
img_width = 128

# Create the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    seed=123,
    validation_split=0.2,
    subset='training',
    image_size=(img_height, img_width),
    batch_size=batch_size,
    labels='inferred',
    shuffle=True
)

# Pull all labels into a flat array
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights_dict = {int(k): round(float(v), 2) for k, v in enumerate(class_weights)}
print("Class Weights:", class_weights_dict)
