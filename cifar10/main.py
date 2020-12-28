import tensorflow as tf

from cifar10.model import Cifar10GAN
from cifar10.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    print(train_x.shape)

    gan = Cifar10GAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4, beta_1=.5),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=train_x / 127.5 - 1,
        epochs=100,
        batch_size=256,
        callback=[
            imshow,
            checkpoint
        ])
