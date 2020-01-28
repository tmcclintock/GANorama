import numpy as np
import time

from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.models as tfkm

class GENERICorama(object):
    """Generic panorama generator.
    """
    def __init__(self, dataset,
                 BATCH_SIZE = 64, test_size = 0.25,
                 latent_dim = 100):
        dataset = np.asarray(dataset)
        assert len(dataset.shape) == 4
        assert dataset.shape[-1] == 3 #3 channels
        self.dimensions = dataset.shape[1:3]
        self.dimensions[0] % 4 == 0
        self.dimensions[1] % 4 == 0
        
        self.dimensions[0] >= 8 == 0
        self.dimensions[1] >= 8 == 0

        assert type(BATCH_SIZE) == int
        self.BATCH_SIZE = BATCH_SIZE

        assert type(latent_dim) == int
        self.latent_dim = latent_dim

        train, test = train_test_split(dataset, test_size = test_size)
        self.train_dataset = tf.data.Dataset.from_tensor_slices(
            train).batch(self.BATCH_SIZE)
        self.test_dataset = tf.data.Dataset.from_tensor_slices(
            test).batch(self.BATCH_SIZE)

        self.create_model()

    def reset_optimizer(self, opt = tfk.optimizers.Adam):
        """Reset the optimizer attached to this generator.

        Args:
            opt (`tensorflow.keras.optimizers`): default is `Adam`

        """
        self.optimizer = opt(1e-4)
        return

    def sample_latent_space(self, n_samples):
        """Samples from the latent space, assuming a
        multivariate standard normal; N(0,1) for each dimension.

        Args:
            n_samples (int): number of samples to make

        Returns:
            tensor of dimensions `n_samples` by `self.latent_dim`

        """
        return tf.random.normal(shape = [n_samples, self.latent_dim])

    def create_model(self):
        """Create the generative model.

        """
        pass

