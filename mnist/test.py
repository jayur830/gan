import numpy as np
import tensorflow as tf
import cv2

epochs = 1000
batch_size = 128
input_dim = 100
num_classes = 10

if __name__ == '__main__':
    kernel_initializer = "he_normal"

    # (1000,) -> (7 * 7 * 128,)
    g = tf.keras.layers.Input(shape=(100,))
    g = tf.keras.layers.Dense(
        units=7 * 7 * 128,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    # (7 * 7 * 128,) -> (7, 7, 128)
    g = tf.keras.layers.Reshape(target_shape=(7, 7, 128))(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    # (7, 7, 128) -> (14, 14, 128)
    g = tf.keras.layers.Conv2DTranspose(
        filters=128,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    # (14, 14, 128) -> (28, 28, 64)
    g = tf.keras.layers.Conv2DTranspose(
        filters=64,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    # (28, 28, 64) -> (28, 28, 32)
    g = tf.keras.layers.Conv2DTranspose(
        filters=32,
        kernel_size=5,
        kernel_initializer=kernel_initializer,
        use_bias=False)(g)
    g = tf.keras.layers.BatchNormalization(momentum=.9)(g)
    g = tf.keras.layers.ReLU()(g)
    # (28, 28, 32) -> (28, 28, 1)
    g = tf.keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=5,
        kernel_initializer=kernel_initializer)(g)
    g = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(g)

    # (100,)
    g_input_z = tf.keras.layers.Input(shape=(input_dim,))
    # (10,)
    g_input_y = tf.keras.layers.Input(shape=(num_classes,))
    g = tf.keras.models.Model([g_input_z, g_input_y], g(tf.keras.layers.Concatenate()([g_input_z, g_input_y])))

    d = tf.keras.layers.LeakyReLU(alpha=.2)
    # (28, 28, 2) -> (14, 14, 32)
    d = tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.LeakyReLU(alpha=.2)(d)
    # (14, 14, 32) -> (7, 7, 64)
    d = tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=5,
        strides=2,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.Flatten()(d)
    d = tf.keras.layers.Dense(
        units=1,
        kernel_initializer=kernel_initializer)(d)
    d = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(d)

    # (28, 28, 1)
    d_input_x = tf.keras.layers.Input(shape=(28, 28, 1))
    # (10,)
    d_input_y = tf.keras.layers.Input(shape=(num_classes,))
    d = tf.keras.models.Model([d_input_x, d_input_y], d(tf.keras.layers.Concatenate()([
        d_input_x,
        tf.keras.layers.Reshape(target_shape=(28, 28, 1))(
            tf.keras.layers.Dense(
                units=28 * 28 * 1,
                kernel_initializer=kernel_initializer)(d_input_y))
    ])))

    d.trainable = False
    d.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)

    gan = tf.keras.models.Model(inputs=[g_input_z, g_input_y], outputs=d(g([g_input_z, g_input_y])))
    gan.compile(
        optimizer=tf.optimizers.RMSprop(learning_rate=1e-4, decay=3e-8),
        loss=tf.losses.binary_crossentropy)

    def imshow(batch, models):
        generator = models[0]
        if batch % 20 == 0:
            gan_output = generator.predict([
                np.random.normal(0, 1, size=(100, input_dim)),
                tf.keras.utils.to_categorical(np.random.randint(num_classes, size=(batch_size, num_classes)))
            ])
            gan_output = gan_output.reshape((gan_output.shape[0],) + (28, 28))
            total_imgs = np.zeros(shape=(0, 800))

            for r in range(10):
                row_imgs = np.zeros(shape=(80, 0))
                for c in range(10):
                    img = (gan_output[r * 10 + c] + 1) / 2
                    img = cv2.copyMakeBorder(
                        src=cv2.resize(
                            src=img,
                            dsize=(60, 60)),
                        top=10,
                        bottom=10,
                        left=10,
                        right=10,
                        borderType=cv2.BORDER_CONSTANT,
                        value=1)
                    row_imgs = np.concatenate([row_imgs, img], axis=1)
                total_imgs = np.concatenate([total_imgs, row_imgs], axis=0)

            cv2.imshow("Test", total_imgs)
            cv2.waitKey(100)

    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    train_x = train_x.reshape(train_x.shape + (1,)) / 255.
    test_x = test_x.reshape(test_x.shape + (1,)) / 255.
    train_y = tf.keras.utils.to_categorical(train_y)
    test_y = tf.keras.utils.to_categorical(test_y)

    batch_count = train_x.shape[0] // batch_size

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        for batch in range(1, batch_count + 1):
            idx = np.random.randint(0, train_x.shape[0], size=batch_size)

            d.trainable = True
            d_loss = d.train_on_batch(
                x=[train_x[idx], train_y[idx]],
                y=np.ones(batch_size))

            z = np.random.normal(0, 1, size=(batch_size, input_dim))
            y = tf.keras.utils.to_categorical(np.random.randint(num_classes, size=(batch_size, num_classes)))
            d_loss += d.train_on_batch(
                x=[g([z, y]), y],
                y=np.zeros(batch_size))

            d_loss /= 2

            z = np.random.normal(0, 1, size=(batch_size, input_dim))
            y = tf.keras.utils.to_categorical(np.random.randint(num_classes, size=(batch_size, num_classes)))
            d.trainable = False
            g_loss = gan.train_on_batch(
                x=[z, y],
                y=np.ones(batch_size))

            print(f"d_loss: {d_loss}, g_loss: {g_loss}")

            imshow(batch, [g, d])
