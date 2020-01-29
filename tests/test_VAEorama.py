import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_VAEorama_smoketest():
    M, N = 8, 8
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data)

def test_attributes():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)
    npt.assert_equal(V.dimensions, [M, N])
    npt.assert_equal(V.latent_dim, LD)
    npt.assert_equal(V.BATCH_SIZE, BS)

def test_training():
    M, N = 16, 128
    LD = 100
    BS = 25
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, latent_dim = LD, BATCH_SIZE = BS)
    V.train(10)

if __name__ == "__main__":
    #test_VAEorama_smoketest()
    #test_attributes()
    test_training()
