import tensorflow as tf
import numpy as np

from mnist.model import MnistGAN
from mnist.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
    x = np.concatenate([x_train, x_test])

    gan = MnistGAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.keras.optimizers.RMSprop(lr=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=x,
        epochs=100,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=imshow),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=checkpoint)
        ])
