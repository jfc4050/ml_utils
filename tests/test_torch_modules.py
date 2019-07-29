import torch

from ml_utils.torch.modules import Normalizer


def test_Normalizer_maintains_shape():

    n = Normalizer()
    x_in = torch.rand(1, 3, 5, 5)
    x_out = n(x_in)

    assert x_in.shape == x_out.shape
