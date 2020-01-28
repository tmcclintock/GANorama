from .generic import GENERICorama

import numpy as np

import tensorflow as tf
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
        
