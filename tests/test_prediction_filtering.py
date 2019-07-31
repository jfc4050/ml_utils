import pytest
import numpy as np

from ml_utils.prediction_filtering import (
    ClasswisePredictionFilter,
    ConfidenceFilter,
    MaxDetFilter,
    NMSFilter,
)


def test_ClasswisePredictionFilter_returns_correct_dets():
    MAX_DETS = 2

    confs = np.random.rand(10)
    classes = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
    bboxes = np.random.rand(10, 4)

    f = ClasswisePredictionFilter(MaxDetFilter(2))

    confs, classes, bboxes = f(confs, classes, bboxes)

    assert (classes == 1).sum() == 2
    assert (classes == 2).sum() == 2


def test_ConfidenceFilter_returns_correct_dets():
    confs = np.array([0.0, 0.5, 1.0])
    bboxes = np.random.rand(len(confs), 4)

    f = ConfidenceFilter(0.75)
    confs, bboxes = f(confs, bboxes)

    assert confs.shape == (1,)
    assert bboxes.shape == (1, 4)


@pytest.mark.parametrize("max_dets", [0, 5, 10])
def test_MaxDetFilter_returns_correct_dets(max_dets):
    N = 50
    confs = np.random.rand(N)
    bboxes = np.random.rand(N, 4)

    f = MaxDetFilter(max_dets)

    confs, bboxes = f(confs, bboxes)

    assert confs.shape == (max_dets,)
    assert bboxes.shape == (max_dets, 4)


def test_NMSFilter_returns_correct_dets():
    confs = np.array([0.5, 1.0])
    bboxes = np.array([[0.5, 0.5, 0.1, 0.1], [0.5, 0.5, 0.1, 0.1]])

    f = NMSFilter(0.5)
    confs, bboxes = f(confs, bboxes)

    assert confs.shape == (1,)
    assert bboxes.shape == (1, 4)
