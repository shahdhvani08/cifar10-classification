
# for the convolutional network
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os, sys
import datetime
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import pandas as pd
from scikeras.wrappers import KerasClassifier

from pathlib import Path
# fetch the root directory to read config

config_dir = (os.path.abspath(Path(__file__).parent.parent.parent.parent) + '/configs/')
sys.path.append(config_dir)
print(config_dir)
import config


# define cnn model
def cnn_model(kernel_size = config.kernel_size,
              pool_size= config.pool_size,
              first_filters = config.first_filters,
              second_filters = config.second_filters,
              third_filters = config.third_filters,
              dropout_conv = config.dropout_conv,
              dropout_dense = config.dropout_dense):

        model = Sequential()
        model.add(Conv2D(first_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(first_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((pool_size)))
        model.add(Dropout(dropout_conv))
        model.add(Conv2D(second_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(second_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((pool_size)))
        model.add(Dropout(dropout_conv))
        model.add(Conv2D(third_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(Conv2D(third_filters, kernel_size, activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(MaxPooling2D((pool_size)))
        model.add(Dropout(dropout_conv))
        model.add(Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dropout(dropout_dense))
        model.add(Dense(10, activation='softmax'))
        # compile model
        opt = SGD(learning_rate=0.001, momentum=0.9)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        return model



checkpoint = ModelCheckpoint(config.MODEL_PATH,
                             monitor='acc',
                             verbose=1, 
                             save_best_only=True,
                             mode='max')

log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# We'll stop training if no improvement after some epochs
earlystopper = EarlyStopping(monitor='val_accuracy', patience=8, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=8, 
                                verbose=1, mode='max', min_lr=0.00001)


callbacks_list = [checkpoint, earlystopper, reduce_lr]


cnn_clf = KerasClassifier(build_fn=cnn_model,
                          batch_size=config.BATCH_SIZE, 
                          validation_split=0.2,
                          epochs=config.EPOCHS,
                          verbose=1,
                          callbacks=callbacks_list
                          )

if __name__ == '__main__':
    
    model = cnn_model()
    model.summary()