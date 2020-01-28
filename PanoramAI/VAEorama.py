from .generic import GENERICorama

import numpy as np

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.keras.callbacks import EarlyStopping
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

    def create_model(self):
        """Create the convolutional variation autoencoder
        model. This function is called automatically by the 
        constructor of the superclass.

        """
        M, N = self.dimensions
        latent_dim = self.latent_dim

        self.encoder = Sequential([
            tfkl.InputLayer(input_shape=(M, N, 3)), 
            tfkl.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2),
                activation='relu', padding="valid"),
            tfkl.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2),
                activation='relu', padding="valid"),
            tfkl.Flatten(),
            #predicting mean and logvar
            tfkl.Dense(latent_dim + latent_dim)
        ])
        
        self.decoder = Sequential([
            tfkl.InputLayer(input_shape=(latent_dim,)),
            tfkl.Dense(units= M * N * 4, activation='relu'),
            tfkl.Reshape(target_shape=(M//4, N//4, 64)),
            tfkl.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'),
            tfkl.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2),
                padding="SAME", activation='relu'),
            tfkl.Conv2DTranspose(
                filters=3, kernel_size=3, strides=(1, 1), padding="SAME",
                activation='sigmoid'),
        ])

