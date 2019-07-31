"""prediction filter objects"""

import abc
from typing import Tuple, Sequence

import numpy as np

from .boundingboxes import get_nms_mask


class PredictionFilter(abc.ABC):
    """used to filter predictions."""

    @abc.abstractmethod
    def __call__(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """filter predictions.

        Args:
            confs: (|B|); unfiltered confidences.
            bboxes: (|B|, 4); unfiltered bounding boxes.

        Returns:
            ndarray: (|B'|,); filtered confidences.
            ndarray: (|B'|, 4); filtered bounding boxes.
        """
        raise NotImplementedError

    @staticmethod
    def _apply_mask(
        mask: np.ndarray, *arrays_to_mask: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, ...]:
        """apply mask to confs, classes, and bboxes."""
        return (a[mask] for a in arrays_to_mask)


class ClasswisePredictionFilter(PredictionFilter):
    """groups predictions by class and independently applies prediction filter to each.
    """

    def __init__(self, prediction_filter: PredictionFilter) -> None:
        self.prediction_filter = prediction_filter

    def __call__(
        self, confs: np.ndarray, classes: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_confs = [np.zeros(0)]
        all_classes = [np.zeros(0)]
        all_bboxes = [np.zeros((0, 4))]

        all_confs, all_classes, all_bboxes = list(), list(), list()
        for cls_id in np.unique(classes):
            cls_confs, cls_bboxes = self._apply_mask(classes == cls_id, confs, bboxes)
            cls_confs, cls_bboxes = self.prediction_filter(cls_confs, cls_bboxes)
            all_confs.append(cls_confs)
            all_classes.append(np.full(len(cls_confs), cls_id))
            all_bboxes.append(cls_bboxes)

        confs = np.concatenate(all_confs)
        classes = np.concatenate(all_classes)
        bboxes = np.concatenate(all_bboxes)

        return confs, classes, bboxes


class PredictionFilterPipeline(PredictionFilter):
    """used to express multiple composed prediction filters as a single
    prediction filtering operation.

    Args:
        *filter_layers: sequence of PredictionFilters to apply.
    """

    def __init__(self, *filter_layers: Sequence[PredictionFilter]) -> None:
        self.filter_layers = filter_layers

    def __call__(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """see superclass."""
        for layer in self.filter_layers:
            confs, bboxes = layer(confs, bboxes)
        return confs, bboxes


class ConfidenceFilter(PredictionFilter):
    """filters based on prediction confidence.

    Args:
        conf_thresh: minimum confidence for prediction to be kept.
    """

    def __init__(self, conf_thresh: float) -> None:
        self.conf_thresh = conf_thresh

    def __call__(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """see superclass."""
        return self._apply_mask(confs > self.conf_thresh, confs, bboxes)


class MaxDetFilter(PredictionFilter):
    """throws out all but top `max_dets` detections."""

    def __init__(self, max_dets: int) -> None:
        self.max_dets = max_dets

    def __call__(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """see superclass."""
        keep_inds = np.argsort(-confs)
        keep_inds = keep_inds[: self.max_dets]

        confs = confs[keep_inds]
        bboxes = bboxes[keep_inds, :]

        return confs, bboxes


class NMSFilter(PredictionFilter):
    """applies non-max suppression.

    Args:
        iou_thresh: minimum iou between two boxes for them to be considered overlapping.
    """

    def __init__(self, iou_thresh: float) -> None:
        self.iou_thresh = iou_thresh

    def __call__(
        self, confs: np.ndarray, bboxes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """see superclass."""
        return self._apply_mask(
            get_nms_mask(confs, bboxes, self.iou_thresh), confs, bboxes
        )
