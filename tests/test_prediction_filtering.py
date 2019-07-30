import pytest
import numpy as np

from ml_utils.prediction_filtering import MaxDetFilter


@pytest.mark.parametrize("max_dets", [0, 5, 10])
def test_MaxDetFilter_returns_correct_dets(max_dets):
    N = 50
    confs = np.random.rand(N)
    classes = np.random.rand(N)
    bboxes = np.random.rand(N, 4)

    f = MaxDetFilter(max_dets)

    confs, classes, bboxes = f(confs, classes, bboxes)

    assert confs.shape == (max_dets,)
    assert classes.shape == (max_dets,)
    assert bboxes.shape == (max_dets, 4)
