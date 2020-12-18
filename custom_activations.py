import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.util.tf_export import keras_export


def thresholded_leaky_relu(alpha, threshold):
    tf.nn.leaky_relu()
    return lambda x: x * alpha if x < 0 else (x if x < threshold else alpha * (x - threshold))


@keras_export("keras.layers.ThresholdedLeakyReLU")
class ThresholdedLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.3, threshold=1., **kwargs):
        super(ThresholdedLeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.alpha = K.cast_to_floatx(alpha)
        self.threshold = K.cast_to_floatx(threshold)

    def call(self, inputs):
        return K.relu(inputs, alpha=self.alpha) if inputs < self.threshold else self.alpha * (K.relu(inputs, alpha=self.alpha) - self.threshold) + self.threshold
