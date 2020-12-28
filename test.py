import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    input_shape = (3, 3, 1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2DTranspose(
            filters=5,
            kernel_size=3,
            input_shape=input_shape)
    ])
    print(model.predict(np.ones(shape=(1,) + input_shape).shape))
