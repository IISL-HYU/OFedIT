import tensorflow as tf
from tensorflow import keras
from keras import layers, models

from .model_custom import CustomModelList, CustomModel


def model_generation(N, metric):
    random_seed = 4
    tf.random.set_seed(random_seed)
  
    all_models = CustomModelList()
    loss_fn = keras.losses.SparseCategoricalCrossentropy()
    for i in range(N):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))
        tf.random.set_seed(random_seed)
        model1 = CustomModel(model)
        model1.compile(optimizer='adam', loss=loss_fn, metrics=metric)
        all_models.append(model1)
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))
    tf.random.set_seed(random_seed)
    central_server = CustomModel(model)
    central_server.compile(optimizer='adam', loss=loss_fn, metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    return all_models, central_server
