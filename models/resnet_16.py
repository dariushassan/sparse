from utils.optimizers.fed_prox import FedProx
import tensorflow as tf
from tensorflow.keras import layers

from models.model import Model


class Resnet16Model(Model):

    def __init__(self, kernel_initializer=Model.InitializationStates.HE_NORMAL, learning_rate=0.02,
                 metrics=["accuracy"], kernel_regularizer=None, bias_regularizer=None,
                 use_sgd=False, use_sgd_with_momentum=False, momentum_factor=0.0,
                 use_fedprox=False, fedprox_mu=0.0):
        super().__init__(kernel_initializer, learning_rate, metrics)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_sgd = use_sgd
        self.use_sgd_with_momentum = use_sgd_with_momentum
        self.momentum_factor = momentum_factor
        self.use_fedprox = use_fedprox
        self.fedprox_mu = fedprox_mu

    def get_model(self):
        # Resnet Architecture
        def residual_stack(x):
            def residual_unit(y,_strides=1):
                shortcut_unit=y
                # 1x1 conv linear
                y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu')(y)
                y = layers.BatchNormalization()(y)
                y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='linear')(y)
                y = layers.BatchNormalization()(y)
                # add batch normalization
                y = layers.add([shortcut_unit,y])
                return y
        
            x = layers.Conv1D(32, data_format='channels_first',kernel_size=1, padding='same',activation='linear')(x)
            x = layers.BatchNormalization()(x)
            x = residual_unit(x)
            x = residual_unit(x)
            # maxpool for down sampling
            x = layers.MaxPooling1D(data_format='channels_first')(x)
            return x

        inputs=layers.Input(shape=[2, 128])
        x = residual_stack(inputs)  # output shape (32,64)
        x = residual_stack(x)    # out shape (32,32)
        x = residual_stack(x)    # out shape (32,16)    # Comment this when the input dimensions are 1/32 or lower
        x = residual_stack(x)    # out shape (32,8)     # Comment this when the input dimensions are 1/16 or lower
        x = residual_stack(x)    # out shape (32,4)     # Comment this when the input dimensions are 1/8 or lower
        x = layers.Flatten()(x)
        x = layers.Dense(128,kernel_initializer="he_normal", activation="selu", name="dense1")(x)
        x = layers.AlphaDropout(0.1)(x)
        x = layers.Dense(128,kernel_initializer="he_normal", activation="selu", name="dense2")(x)
        x = layers.AlphaDropout(0.1)(x)
        x = layers.Dense(10,kernel_initializer="he_normal", activation="softmax", name="dense3")(x)
        x_out = layers.Reshape([10])(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=x_out)

        if self.use_sgd:
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0),
                loss="categorical_crossentropy", metrics=self.metrics)
        if self.use_sgd_with_momentum:
            if self.momentum_factor == 0.0:
                raise RuntimeError("Need to provide a non-zero value for the momentum attenuation term.")
            # So far we have run experiments with m=0.9.
            model.compile(
                optimizer=tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum_factor),
                loss="categorical_crossentropy", metrics=self.metrics)
        if self.use_fedprox:
            if self.fedprox_mu == 0.0:
                raise RuntimeError("Need to provide a non-zero value for the FedProx proximal term.")
            # So far we have run experiments with μ=0.001.
            model.compile(
                optimizer=FedProx(learning_rate=self.learning_rate, mu=self.fedprox_mu),
                loss="categorical_crossentropy", metrics=self.metrics)

        return model