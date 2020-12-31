import tensorflow as tf

from cifar10.model import Cifar10GAN
from cifar10.callbacks import imshow, checkpoint

if __name__ == '__main__':
    (train_x, _), (_, _) = tf.keras.datasets.cifar10.load_data()
    print(train_x.shape)

    gan = Cifar10GAN(input_shape=(100,))
    gan.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=train_x / 127.5 - 1,
        epochs=100,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=imshow),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=checkpoint)
        ])
