import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from lpr.model import LprGAN
from lpr.callbacks import imshow, checkpoint


if __name__ == '__main__':
    x = []
    imgs = glob("./lane_day_ag_1/*.jpg")
    for img in imgs:
        x.append(cv2.resize(cv2.imread(img), dsize=(640, 368), interpolation=cv2.INTER_AREA))
    x = np.asarray(x).astype("float32") / 255.

    gan = LprGAN(input_shape=(256,))
    gan.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)
    gan.fit(
        x=x,
        epochs=100000,
        batch_size=2,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_batch_end=imshow),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=checkpoint)
        ])