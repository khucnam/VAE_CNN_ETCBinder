import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, *args, **kwargs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEModel(keras.Model):
    def __init__(self, encoder, decoder, alpha, **kwargs):
        super(VAEModel, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.alpha = alpha

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.mean_squared_error(data, reconstruction), axis=-1
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss*self.alpha + kl_loss*(1-self.alpha)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


class VAE:
    def __init__(self, latent_dim, alpha, lr, epoch, batch_size, stay_persist=True):
        self.input_shape = (300, 1)
        self.num_channels = 1
        self.latent_dim = latent_dim
        self.alpha=alpha
        self.optimizer = keras.optimizers.Adam(lr)
        self.epoch=epoch
        self.batch_size = batch_size
        self.model = VAEModel(
            encoder=self.build_encoder(),
            decoder=self.build_decoder(),
            alpha=self.alpha
        )
        # print("VAE .....",self.model.summary())

        self.model.compile(
            optimizer=self.optimizer
        )

        # dir_path = os.path.dirname(os.path.realpath(__file__))
        # init_path = os.path.join(dir_path, "..", "pretrained", "init.h5")
        # if not os.path.isfile(init_path) and stay_persist:
        #     self.model.save(init_path)
        #
        # if stay_persist:
        #     self.model = keras.models.load_model(init_path)

    def build_encoder(self):
        encoder_inputs = keras.layers.Input(shape=self.input_shape, name="encoder_input")
        encoder_conv_layer1 = layers.Conv1D(filters=128, kernel_size=16, padding="same", strides=2,
                                                  name="encoder_conv_1")(encoder_inputs)
        encoder_norm_layer1 = layers.BatchNormalization(name="encoder_norm_1")(encoder_conv_layer1)
        encoder_activ_layer1 = layers.LeakyReLU(name="encoder_leakyrelu_1")(encoder_norm_layer1)
        encoder_conv_layer2 = layers.Conv1D(filters=64, kernel_size=8, padding="same", strides=1,
                                                  name="encoder_conv_2")(encoder_activ_layer1)
        encoder_norm_layer2 = layers.BatchNormalization(name="encoder_norm_2")(encoder_conv_layer2)
        encoder_activ_layer2 = layers.LeakyReLU(name="encoder_activ_layer_2")(encoder_norm_layer2)
        encoder_conv_layer3 = layers.Conv1D(filters=32, kernel_size=8, padding="same", strides=2,
                                                  name="encoder_conv_3")(encoder_activ_layer2)
        encoder_norm_layer3 = layers.BatchNormalization(name="encoder_norm_3")(encoder_conv_layer3)
        encoder_activ_layer3 = layers.LeakyReLU(name="encoder_activ_layer_3")(encoder_norm_layer3)

        encoder_flatten = keras.layers.Flatten()(encoder_activ_layer3)
        print(encoder_flatten.shape)
        z_mean = layers.Dense(self.latent_dim, name="z_mean")(encoder_flatten)
        z_log_var = layers.Dense(self.latent_dim, name="z_log_var")(encoder_flatten)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
        print("Encoder ......", encoder.summary())

        return encoder

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,), name="decoder_input")
        decoder_dense_layer1 = layers.Dense(units=75 * 32, name="decoder_dense_1")(
            latent_inputs)
        decoder_reshape = layers.Reshape(target_shape=(75, 32))(decoder_dense_layer1)
        decoder_conv_tran_layer1 = layers.Conv1DTranspose(filters=32, kernel_size=8, padding="same", strides=1,
                                                                name="decoder_conv_tran_1")(decoder_reshape)
        decoder_norm_layer1 = layers.BatchNormalization(name="decoder_norm_1")(decoder_conv_tran_layer1)
        decoder_activ_layer1 = layers.LeakyReLU(name="decoder_leakyrelu_1")(decoder_norm_layer1)
        decoder_conv_tran_layer2 = layers.Conv1DTranspose(filters=64, kernel_size=8, padding="same", strides=2,
                                                                name="decoder_conv_tran_2")(decoder_activ_layer1)
        decoder_norm_layer2 = layers.BatchNormalization(name="decoder_norm_2")(decoder_conv_tran_layer2)
        decoder_activ_layer2 = layers.LeakyReLU(name="decoder_leakyrelu_2")(decoder_norm_layer2)
        decoder_conv_tran_layer3 = layers.Conv1DTranspose(filters=128, kernel_size=8, padding="same", strides=1,
                                                                name="decoder_conv_tran_3")(decoder_activ_layer2)
        decoder_norm_layer3 = keras.layers.BatchNormalization(name="decoder_norm_3")(decoder_conv_tran_layer3)
        decoder_activ_layer3 = keras.layers.LeakyReLU(name="decoder_leakyrelu_3")(decoder_norm_layer3)
        decoder_conv_tran_layer4 = keras.layers.Conv1DTranspose(filters=1, kernel_size=16, padding="same", strides=2,
                                                                name="decoder_conv_tran_4")(decoder_activ_layer3)
        decoder_output = keras.layers.LeakyReLU(name="decoder_output")(decoder_conv_tran_layer4)
        decoder = keras.models.Model(latent_inputs, decoder_output, name="decoder_model")
        print("decoder ......", decoder.summary())
        return decoder

    def fit(self, train_X):
        self.model.fit(train_X, epochs=self.epoch, batch_size=self.batch_size)

    def sampling(self, X):
        return self.model.encoder.predict(X)[-1]

    def generate(self, X):
        z = self.model.encoder.predict(X)[-1]

        return self.model.decoder.predict(z)
