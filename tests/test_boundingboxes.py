import pytest
import numpy as np

from ml_utils import boundingboxes


@pytest.mark.parametrize("n_boxes", [0, 1, 2])
def test_ijhw_to_ijij_doesnt_change_shape(n_boxes):
    bboxes_ijhw = np.random.rand(n_boxes, 4)
    bboxes_ijij = boundingboxes.ijhw_to_ijij(bboxes_ijhw)

    assert bboxes_ijhw.shape == bboxes_ijij.shape


def test_ijhw_to_ijij_keeps_values_inbounds():
    bboxes_ijhw = np.array([[0.5, 0.5, 1.5, 1.5]])
    bboxes_ijij = boundingboxes.ijhw_to_ijij(bboxes_ijhw)

    assert (bboxes_ijij < 0).sum() == 0
    assert (bboxes_ijij > 1).sum() == 0


@pytest.mark.parametrize('n_boxes_a', [0, 1, 2])
@pytest.mark.parametrize('n_boxes_b', [0, 1, 2])
def test_compute_ious_gives_correct_shapes(n_boxes_a, n_boxes_b):
    bboxes_a = np.random.rand(n_boxes_a, 4)
    bboxes_b = np.random.rand(n_boxes_b, 4)

    ious = boundingboxes.compute_ious(bboxes_a, bboxes_b)

    assert ious.shape == (n_boxes_a, n_boxes_b)
