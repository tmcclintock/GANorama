import PanoramAI as PA
import numpy as np
import numpy.testing as npt
import pytest

def test_GENERICorama():
    #Smoke test
    data = np.random.randn(10, 4, 8, 3)
    o = PA.GENERICorama(data)

if __name__ == "__main__":
    test_GENERICorama()
