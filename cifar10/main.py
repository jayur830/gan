import tensorflow as tf
import numpy as np

from cifar10.model import Cifar10GAN
from cifar10.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    print(train_x.shape)

    gan = Cifar10GAN(input_dim=100)
    gan.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    history = gan.fit(
        x=train_x / 127.5 - 1.,
        epochs=5000,
        batch_size=128,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=imshow),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=checkpoint)
        ])
    np.save(file="./history.npy", arr=np.asarray(history))
