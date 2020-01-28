import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_VAEorama():
    #Smoke test
    M, N = 16, 128
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, M, N)

def test_training():
    M, N = 16, 128
    data = np.random.randn(128, M, N, 3)
    V = PA.VAEorama(data, M, N)
    V.train(10)

if __name__ == "__main__":
    test_training()
