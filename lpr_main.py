import cv2
import numpy as np
import tensorflow as tf

from glob import glob
from lpr_gan import LprGAN
from callbacks import lpr_imshow


if __name__ == '__main__':
    x = []
    imgs = glob("./lane_day_ag_1/*.jpg")
    for img in imgs:
        x.append(cv2.resize(cv2.imread(img), dsize=(160, 92), interpolation=cv2.INTER_AREA))
    x = np.asarray(x).astype("float32") / 255.

    gan = LprGAN(input_shape=(23 * 40,))
    gan.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-4, beta_1=.5),
        loss=tf.losses.mean_absolute_error)
    gan.fit(
        x=x,
        epochs=100,
        batch_size=16,
        callback=[lpr_imshow])