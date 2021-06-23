import tensorflow as tf

from keras.activations import relu
from keras.layers import Conv2D, BatchNormalization


class Residual_nlayer(tf.keras.Model):
    def __init__(self, num_layers, num_channels, strides=1, use_1x1=False):
        super().__init__()

        self.conv0 = Conv2D(num_channels, padding='same', kernel_size=3, strides=strides)
        self.bn0 = BatchNormalization()
        self.inner_layers = []

        for _ in range(num_layers - 1):
            self.inner_layers.append(Conv2D(num_channels, kernel_size=3, padding='same'))
            self.inner_layers.append(BatchNormalization())

        self.convSkip = None
        if use_1x1:
            self.convSkip = Conv2D(num_channels, kernel_size=1, strides=strides)

    def call(self, X):
        Y = self.conv0(X)
        Y = self.bn0(Y)
        for idx, layer in enumerate(self.inner_layers):
            if idx % 2 == 0:
                Y = relu(Y)
            Y = layer(Y)

        if self.convSkip:
            X = self.convSkip(X)

        return relu(Y + X)


class GeneralResnetBlock(tf.keras.layers.Layer):
    def __init__(self, num_layers, num_channels, num_residuals, first_block=False, **kwargs):
        super(GeneralResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(Residual_nlayer(num_layers, num_channels, use_1x1=True, strides=2))
            else:
                self.residual_layers.append(Residual_nlayer(num_layers, num_channels))

    def call(self, X):
        for layer in self.residual_layers.layers:
            X = layer(X)
        return X