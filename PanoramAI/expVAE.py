import numpy as np
import time

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from tensorflow.keras import backend as K

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential

class VAEorama(object):
    def __init__(self, dataset, latent_dim = 200):
        dataset = np.asarray(dataset)
        assert len(dataset.shape) == 4
        assert dataset.shape[-1] == 3 #3 channels
        assert type(latent_dim) == int
        
        self.latent_dim = 200
        self.dimensions = dataset.shape[1:3]
        self.dimensions[0] % 4 == 0
        self.dimensions[1] % 4 == 0

        self.dataset = dataset
        self.batch_size = 64

        self.reset_optimizer()
        self.create_CVAE()

    def reset_optimizer(self, opt = tf.keras.optimizers.Adam):
        self.optimizer = opt(1e-4)
        return

    """
    def _generate_random_vector(self, n_samples):
        self.n_samples_to_generate = n_samples
        self.random_vector_for_generation = tf.random.normal(
            shape=[n_samples, self.latent_dim])
        return

    def generate_samples(self, n_samples):
        if n_samples > self.n_samples_to_generate:
            print("Regenerating sample vector.")
            self._generate_random_vector(n_samples)
        return self.CVAE.sample(self.random_vector_for_generation)
    """


    """
    def log_normal_pdf(self, sample, mean, logvar):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi))
    """
    """
    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.CVAE.encode(x)
        z = self.CVAE.reparameterize(mean, logvar)
        x_predicted = self.CVAE.decode(z)
        MSE = tf.losses.MSE(x, x_predicted)
        logpx_z = -tf.reduce_sum(MSE)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)
    """

    #@tf.function
    #def compute_apply_gradients(self, x):
    #    with tf.GradientTape() as tape:
    #        loss = self.compute_loss(x)
    #        gradients = tape.gradient(loss, self.CVAE.trainable_variables)
    #        self.optimizer.apply_gradients(
    #            zip(gradients, self.CVAE.trainable_variables))

    def train(self, epochs):
        callback = EarlyStopping(monitor='val_loss', patience=3)
        self.CVAE.fit(x = self.dataset,
                      batch_size = self.batch_size,
                      epochs = epochs,
                      callbacks = [callback,], validation_split = 0.3)

    def _reparameterize(self, args):
        """Reparameterization trick by sampling from an isotropic unit Gaussian.
        
        Arguments
            args (tensor): mean and log of variance of Q(z|X)
        
        Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    #@tf.function
    #def sample(self, eps=None):
    #    if eps is None:
    #        eps = tf.random.normal(shape=(100, self.latent_dim))
    #    return self.decoder(eps)
    
    def create_CVAE(self):
        M, N = self.dimensions
        ld = self.latent_dim

        #encoder model
        inputs = tfkl.Input(shape = (M, N, 3),
                            batch_size = self.batch_size,
                            name = 'encoder_input')
        x = tfkl.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2),
                activation='relu', padding="valid")(inputs)
        x = tfkl.Conv2D(
                filters=64, kernel_size=3,strides=(2, 2),
                activation='relu', padding="valid")(x)
        x = tfkl.Flatten()(x)
        z_mean = tfkl.Dense(units = ld, name='z_mean')(x)
        z_log_var = tfkl.Dense(units = ld, name='z_log_var')(x)

        #Reparameterization trick
        z = tfkl.Lambda(self._reparameterize,
                        output_shape=(ld,), name='z')([z_mean, z_log_var])
        
        #Instantiate the encoder
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        #decoder model
        latent_inputs = tfkl.Input(shape=(ld,), name='z_sampling')
        x = tfkl.Dense(units = M * N * 4, activation='relu')(latent_inputs)
        x = tfkl.Reshape(target_shape=(M//4, N//4, 64))(x)
        x = tfkl.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2),
            padding="SAME", activation='relu')(x)
        x = tfkl.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2),
            padding="SAME", activation='relu')(x)
        outputs = tfkl.Conv2DTranspose(
            filters=3, kernel_size=3, strides=(1, 1), padding="SAME",
            activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])
        vae = Model(inputs, outputs, name='cvae')
        
        #Make the loss
        reconstruction_loss = tfk.losses.mse(inputs, outputs)
        reconstruction_loss = -tf.reduce_sum(reconstruction_loss)
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        self.CVAE = vae
        self.encoder = encoder
        self.decoder = decoder
        """
        encoder = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(M, N, 3)),#(bs, M, N, 3)
            tfkl.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2),
                activation='relu', padding="valid"), #(bs, M/2, N/2, 32)
            tfkl.Conv2D(
                filters=64, kernel_size=3,strides=(2, 2),#(bs, M/4, N/4, 64)
                activation='relu', padding="valid"),
            tfkl.Flatten(), #(bs, (M/4) * (N/4) * 64)
            #predicting mean and logvar
            tfkl.Dense(ld + ld), # (bs, D * D)
        ])
        decoder  = tf.keras.Sequential([
            tfkl.InputLayer(input_shape=(ld,)), #(bs, D)
            #M * N * 4 = (M/4)*(N/4)*64
            tfkl.Dense(units= M * N * 4, activation=tf.nn.relu), 
            tfkl.Reshape(target_shape=(M//4, N//4, 64)),
            tfkl.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'), #(bs, M/2, N/2, 64)
            tfkl.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'), #(bs, M, N, 32)
            tfkl.Conv2DTranspose(
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME",
                activation='sigmoid'), #(bs, M, N, 3)
        ])
        """
        #self.CVAE = Model(inputs = encoder.inputs,
        #                  outputs = decoder(encoder.outputs[0]))
        #self.CVAE.add_loss(self.compute_loss)
        #self.CVAE.compile(
        #    optimizer = self.optimizer, loss = None)
