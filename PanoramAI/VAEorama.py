from .generic import GENERICorama

import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import RandomNormal

class VAEorama(GENERICorama):
    """Variational autoencoder (VAE) used to learn panoramic images.
    Specifically, the VAE is convolutional (ConVAE), and is in
    a `_CVAE` object attribute.

    The `VAEorama` contains the routines for training the 
    networks and for generating sample images.

    Args:
        M (int): pixel height of the input images
        N (int): pixel width of the input images
        latent_dim (int): size of the latent space
        n_samples_to_generate (int): number of samples
            to automatically generate when doing random
            sample generation
        optimizer (`tf.keras.optimizers`): default is Adam(1e-4)
        train_dataset (`numpy.ndarray`): input training image dataset
        test_dataset (`numpy.ndarray`): input testing image dataset
        BATCH_SIZE (int): batch size for training

    """
    def __init__(self, dataset,
                 BATCH_SIZE = 64, test_size = 0.25,
                 latent_dim = 100):
        super().__init__(dataset, BATCH_SIZE, test_size, latent_dim)

    def _log_normal_pdf(self, sample, mean, logvar, raxis=1):
        """Log PDF of the multivariate normal distribution
        given tensors of means and log(variances). Note that
        the element-wise log-pdf is reduced along `raxis`.

        Args:
            sample (`tensor`): random variable
            mean (`tensor`): mean of distributions
            logvar (`tensor`): log of the variances
            raxis (int): axis to take the sum over; default 1
        
        Returns:
            (`float`): log pdf of the sample

        """
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    """
    def VAE_loss(self):
        # KL loss - how to pass in x???
        mean, logvar = tf.split(self.encoder(x), 
                                num_or_size_splits=2, axis=1)
        #Sample from a multivariate Gaussian
        z = tf.random.normal(shape=mean.shape) * tf.exp(logvar * .5) + mean
        logpz = self._log_normal_pdf(z, 0., 0.)
        logqz_x = self._log_normal_pdf(z, mean, logvar)

        def total_loss(y_true, y_pred):
            # Reconstruction loss
            MSE = tf.losses.MSE(y_true, y_pred)
            logpx_z = -tf.reduce_sum(MSE)
            return -tf.reduce_mean(logpx_z + logpz - logqz_x)

        return total_loss
        """

    def create_model(self):
        """Create the convolutional variation autoencoder
        model. This function is called automatically by the 
        constructor of the superclass.

        """
        M, N = self.dimensions
        latent_dim = self.latent_dim
        input_shape = (M, N, 3)

        encoder_input = tfkl.Input(shape = input_shape)
        X = Conv(32, 3)(encoder_input)
        X = Conv(64, 3)(X)
        X = tfkl.Flatten()(X)
        #X = tfkl.Dense(latent_dim + latent_dim)(X)
        Z_mu = tfkl.Dense(latent_dim)(X)
        Z_logvar = tfkl.Dense(latent_dim, activation="relu")(X)
        Z = Reparameterize()([Z_mu, Z_logvar])

        decoder_input = tfkl.Input(shape = (latent_dim,))
        X = tfkl.Dense(M * N * 4,  activation="relu")(decoder_input)
        X = tfkl.Reshape(target_shape = (M//4, N//4, 64))(X)
        X = Deconv(64, 3)(X)
        X = Deconv(32, 3)(X)
        decoder_output = Deconv(3, 3, strides=1, activation="sigmoid")(X)

        def reconstruction_loss(X, X_pred):
            mse = tf.losses.MeanSquaredError()
            return mse(X, X_pred) * np.prod(input_shape)

        #KL divergence between a unit normal Gaussian (the prior; p(z)) and q(z|x)
        def kl_divergence(X, X_pred):
            #self.C += (1/1440) # TODO use correct scalar
            #self.C = min(self.C, 35) # TODO make variable
            kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu**2 - tf.math.exp(Z_logvar))
            return tf.math.abs(kl)#self.gamma * tf.math.abs(kl - self.C)

        def total_loss(X, X_pred):
            return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)

        self.encoder = Model(encoder_input, [Z_mu, Z_logvar, Z])
        self.decoder = Model(decoder_input, decoder_output)
        self.vae = Model(encoder_input, self.decoder(Z))
        self.vae.compile(optimizer=self.optimizer,
                         loss=total_loss,
                         metrics=[reconstruction_loss, kl_divergence])

    def train(self, epochs, steps_for_update = None, quiet = False):
        """Train the networks in the convolutional VAE.

        Args:
            epochs (int): number of epochs
            steps_for_update (int): number of epochs to
                compute before giving a status update
            quiet (bool): whether to give status updates

        """
        if not steps_for_update:
            steps_for_update = epochs // 10

        start_time = time.time()
        for epoch in range(1, epochs + 1):
            for train_x in self.train_dataset:
                self.compute_apply_gradients(train_x)

            if epoch % steps_for_update == 0:
                end_time = time.time()
                loss = tf.keras.metrics.Mean()
                if self.test_dataset is not None:
                    for test_x in self.test_dataset:
                        loss(self.compute_loss(test_x))
                elbo = -loss.result()
                #self.TESTING_LOSS.append(elbo)
                #self.RECORDED_EPOCHS.append(epoch)

                if not quiet:
                    print(f'Epoch: {epoch}, Test set ELBO: {elbo:.4f}, '
                          f'time elapsed for current epoch batch {end_time - start_time:.4f}')
                start_time = time.time()
            if epoch == epochs:
                break
        #self.TOTAL_EPOCHS += epochs
        #if not quiet:
        #    print(f"Total epochs: {self.TOTAL_EPOCHS}")
        return


def Conv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return tfkl.Conv2D(n_filters, filter_width, 
                       strides=strides, padding="valid",
                       activation=activation, name=name)


def Deconv(n_filters, filter_width, strides=2, activation="relu", name=None):
    return tfkl.Conv2DTranspose(n_filters, filter_width, 
                                strides=strides, padding="same",
                                activation=activation, name=name)

class Reparameterize(tfkl.Layer):
    """
    Custom layer.
     
    Reparameterization trick, sample random latent vectors Z from 
    the latent Gaussian distribution which has the following parameters 

    mean = Z_mu
    logvar = Z_logvar
    """
    def call(self, inputs):
        Z_mu, Z_logvar = inputs
        return Z_mu + tf.math.exp(0.5 * Z_logvar) * tf.random.normal(tf.shape(Z_mu))
