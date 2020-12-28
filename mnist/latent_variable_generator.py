import tensorflow as tf
import numpy as np


class MnistGANDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x, batch_size=32, shuffle=True):
        self.__x = x
        self.__batch_size = batch_size
        self.__shuffle = shuffle
        self.__indexes = np.arange(0, self.__x.shape[0])

    def __getitem__(self, item):
        pass

    def __len__(self):
        return self.__x.shape[0] // self.__batch_size

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indexes)
