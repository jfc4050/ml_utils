"""prediction filter objects"""

import abc

import numpy as np

from .boundingboxes import get_nms_mask


class PredictionFilter(abc.ABC):
    """used to filter confidences, classes, and boxes."""

    @abc.abstractmethod
    def __call__(self, confs, classes, bboxes):
        """filter confs, classes, and boxes.

        Args:
            confs (ndarray): (nP); unfiltered confs.
            classes (ndarray): (nP); unfiltered classes.
            bboxes (ndarray): (nP, 4); unfiltered bboxes.

        Returns:
            ndarray: (nP'); filtered confs.
            ndarray: (nP', n_classes); filtered classes.
            ndarray: (nP', 4); filtered bboxes.
        """
        raise NotImplementedError

    @staticmethod
    def _apply_mask(mask, *arrays_to_mask):
        """apply mask to confs, classes, and bboxes"""
        return [a[mask] for a in arrays_to_mask]


class PredictionFilterPipeline(PredictionFilter):
    """used to express multiple composed prediction filters as a single
    prediction filtering operation.

    Args:
        *filter_layers ([PredictionFilter]): list of PredictionFilters to apply

    Attributes:
        filter_layers ([PredictionFilter]): see Args.
    """

    def __init__(self, *filter_layers):
        self.filter_layers = filter_layers

    def __call__(self, confs, classes, bboxes):
        """see superclass."""
        for layer in self.filter_layers:
            confs, classes, bboxes = layer(confs, classes, bboxes)
        return confs, classes, bboxes


class ConfidenceFilter(PredictionFilter):
    """filters based on prediction confidence threshold.

    Args:
        conf_thresh (float): minimum confidence for prediction to be kept.

    Attributes:
        conf_thresh (float): see Args.
    """

    def __init__(self, conf_thresh):
        self.conf_thresh = conf_thresh

    def __call__(self, confs, classes, bboxes):
        """see superclass."""
        conf_mask = confs > self.conf_thresh
        return self._apply_mask(conf_mask, confs, classes, bboxes)


class NMSFilter(PredictionFilter):
    """applies class-wise non-max suppression.

    Args:
        iou_thresh (float): minimum iou between two boxes for them to be
            considered overlapping.

    Attributes:
        iou_thresh (float): see Args.
    """

    def __init__(self, iou_thresh):
        self.iou_thresh = iou_thresh

    def __call__(self, confs, classes, bboxes):
        """see superclass."""
        filtered_confs = [np.empty(0, dtype="float32")]
        filtered_classes = [np.empty(0, dtype="int32")]
        filtered_bboxes = [np.empty((0, 4), dtype="float32")]

        for class_id in np.unique(classes):
            ### all predictions of class class_id
            cls_mask = classes == class_id
            class_confs, class_boxes = self._apply_mask(cls_mask, confs, bboxes)

            ### perform classwise nms
            nms_mask = get_nms_mask(class_confs, class_boxes)
            class_confs, class_boxes = self._apply_mask(
                nms_mask, class_confs, class_boxes
            )

            filtered_confs.append(class_confs)
            filtered_classes.append(np.ones(len(class_confs), dtype="int32") * class_id)
            filtered_bboxes.append(class_boxes)

        confs = np.concatenate(filtered_confs)
        classes = np.concatenate(filtered_classes)
        bboxes = np.concatenate(filtered_bboxes)

        return confs, classes, bboxes
