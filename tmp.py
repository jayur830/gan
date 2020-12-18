import tensorflow as tf

if __name__ == '__main__':
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(train_x.shape + (1,)).astype("float32") / 255.
    test_x = test_x.reshape(test_x.shape + (1,)).astype("float32") / 255.
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)

    alpha, threshold = .01, 1.
    model = tf.keras.models.Sequential([
        # (28, 28, 1) -> (28, 28, 8)
        tf.keras.layers.Conv2D(
            filters=8,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False,
            input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(lambda x: max(0, x)),
        # (28, 28, 8) -> (14, 14, 8)
        tf.keras.layers.MaxPool2D(),
        # (14, 14, 8) -> (14, 14, 16)
        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(lambda x: max(0, x)),
        # (14, 14, 16) -> (7, 7, 16)
        tf.keras.layers.MaxPool2D(),
        # (7, 7, 16) -> (7, 7, 32)
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            padding="same",
            kernel_initializer="he_normal",
            use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation(lambda x: max(0, x)),
        # Global Max Pooling
        tf.keras.layers.GlobalMaxPool2D(),
        tf.keras.layers.Dropout(rate=.3),
        tf.keras.layers.Dense(
            units=10,
            activation="softmax",
            kernel_initializer="he_normal")
    ])
    model.compile(
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=.001,
            momentum=.9,
            nesterov=False),
        loss=tf.losses.categorical_crossentropy,
        metrics=["acc"])
    model.summary()

    model.fit(
        x=train_x,
        y=train_y,
        epochs=20,
        batch_size=256,
        validation_split=.2)
