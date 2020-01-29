from .generic import GENERICorama

import numpy as np
import time

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential

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

    def _generate_random_vector(self, n_samples):
        self.n_samples_to_generate = n_samples
        self.random_vector_for_generation = tf.random.normal(
            shape=[n_samples, self.latent_dim])
        return

    def generate_samples(self, n_samples):
        if n_samples > self.n_samples_to_generate:
            print("Regenerating sample vector.")
            self._generate_random_vector(n_samples)
        return self.model.sample(self.random_vector_for_generation)

    def create_model(self):
        M, N = self.dimensions
        self.model = _CVAE(M, N, self.latent_dim)
        return

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.model.encode(x)
        z = self.model.reparameterize(mean, logvar)
        x_predicted = self.model.decode(z)
        MSE = tf.losses.MSE(x, x_predicted)
        logpx_z = -tf.reduce_sum(MSE)
        logpz = self.log_normal_pdf(z, 0., 0.)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables))

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

                if not quiet:
                    print(f'Epoch: {epoch}, Test set ELBO: {elbo:.4f}, '
                          f'time elapsed for current epoch batch {end_time - start_time:.4f}')
                start_time = time.time()
            if epoch == epochs:
                break
        return

    def save_model_weights(self, path = "/tmp/weights/"):
        """Save the VAE network weights.

        Args:
            path (string): to directory where the network 
            weights are saved

        """
        self.model.inference_net.save_weights(
            path + "inference_net_weights", save_format = 'tf')
        self.model.generative_net.save_weights(
            path + "generatives_net_weights", save_format = 'tf')
        return

    def load_model_weights(self, path = "/tmp/weights/"):
        """Load the VAE network weights.

        Args:
            path (string): to directory where the network 
            weights are saved

        """
        self.model.inference_net.load_weights(
            path + "inference_net_weights")
        self.model.generative_net.load_weights(
            path + "generatives_net_weights")
        return

class _CVAE(tf.keras.Model):
    """A convolutional variational autoencoder used to
    create panoramic images.

    Note: we assume there are 3 input (RGB) channels.

    """
    def __init__(self, M, N, latent_dim):
        super(_CVAE, self).__init__()        
        self.input_dims = [M, N]
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(M, N, 3)), #(bs, M, N, 3)
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2),
                activation='relu', padding="valid"), #(bs, M/2, N/2, 32)
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), #(bs, M/4, N/4, 64)
                activation='relu', padding="valid"),
            tf.keras.layers.Flatten(), #(bs, (M/4) * (N/4) * 64)
            #predicting mean and logvar
            tf.keras.layers.Dense(latent_dim + latent_dim), # (bs, D * D)
        ])
        self.generative_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)), #(bs, D)
            tf.keras.layers.Dense(units= M * N * 4, activation=tf.nn.relu), #M * N * 4 = (M/4)*(N/4)*64
            tf.keras.layers.Reshape(target_shape=(M//4, N//4, 64)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'), #(bs, M/2, N/2, 64)
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'), #(bs, M, N, 32)
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME",
                activation='sigmoid'), #(bs, M, N, 3)
        ])
        
    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), 
                                num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        return self.generative_net(z)
