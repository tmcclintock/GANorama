import numpy as np

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
        
        #Attributes to track loss
        self.BEST_LOSS = -1e99

        self.reset_optimizer()
        self.create_model()

    def reset_optimizer(self, opt = tfk.optimizers.Adam):
        """Reset the optimizer attached to this generator.

        Args:
            opt (`tensorflow.keras.optimizers`): default is `Adam`

        """
        self.optimizer = opt(1e-4)
        return

    def save_model(self, epoch, loss, recon, kl, save_path = "./models/"):
        """Write logs and save the model"""
        train_summary_writer = tf.summary.create_file_writer(summary_path)
        with train_summary_writer.as_default():
            tf.summary.scalar("Total Loss", loss, step=epoch)
            tf.summary.scalar("KL Divergence", kl, step=epoch)
            tf.summary.scalar("Reconstruction Loss", recon, step=epoch)

        # save model
        if loss < self.BEST_LOSS:
            self.BEST_LOSS = loss
            self.model.save(save_path+"BEST_MODEL")
        self.model.save(save_path)
        
    def create_model(self):
        """Create the generative model.

        """
        pass

