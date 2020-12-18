import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras.utils import tf_utils


@keras_export("keras.layers.ThresholdedLeakyReLU")
class ThresholdedLeakyReLU(tf.keras.layers.Layer):
    def __init__(self, alpha=0.3, threshold=1., **kwargs):
        super(ThresholdedLeakyReLU, self).__init__(**kwargs)
        self.supports_masking = True
        self.__alpha = K.cast_to_floatx(alpha)
        self.__threshold = K.cast_to_floatx(threshold)

    def call(self, x, **kwargs):
        return K.relu(x, alpha=self.__alpha) if x < self.__threshold else self.__alpha * (x + self.__threshold) + self.__threshold

    def get_config(self):
        return dict(list(super(ThresholdedLeakyReLU, self).get_config().get.items()) + list({ "alpha": self.__alpha}.items()))

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        return input_shape
