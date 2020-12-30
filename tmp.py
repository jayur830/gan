import tensorflow as tf


if __name__ == '__main__':
    print(tf.keras.callbacks.LambdaCallback(on_batch_end=lambda _, logs: print(logs)))
