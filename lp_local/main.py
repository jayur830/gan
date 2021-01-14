import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from lp_local.model import LPLocalGAN
from lp_local.callbacks import imshow, checkpoint


if __name__ == '__main__':
    x = []
    imgs = glob("./WHITE_TWO_LINE_LOCAL/*.jpg")
    for img in imgs:
        x.append(cv2.resize(cv2.imread(img), dsize=(48, 24), interpolation=cv2.INTER_AREA))
    x = np.asarray(x).astype("float32") / 255.

    gan = LPLocalGAN(input_shape=(100,))
    gan.compile_discriminator(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=5e-5, decay=6e-8),
        loss=tf.losses.binary_crossentropy)
    gan.compile_gan(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=x,
        epochs=100000,
        batch_size=10,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=imshow),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=checkpoint)
        ])