"""object detection utilities"""

import numpy as np


def ijhw_to_ijij(ijhw):
    """convert: ijhw->ijij in fractional coordinates.

    Args:
        ijhw (ndarray): (N, 4); ijhw boxes, fractional coordinates.

    Returns:
        ndarray: (N, 4); ijij boxes, fractional coordinates.
    """
    i0j0 = ijhw[:, :2] - ijhw[:, 2:] / 2
    i1j1 = ijhw[:, :2] + ijhw[:, 2:] / 2
    ijij = np.concatenate([i0j0, i1j1], axis=1)
    ijij = np.clip(ijij, 0, 1)  # enforce bounds (relies on fractional coords)
    return ijij


def compute_ious(col_bboxes, row_bboxes):
    """compute intersection over union for all bounding box pairs in cartesian
    product of a_boxes and b_boxes.

    Args:
        col_bboxes (ndarray): (nA, 4); ijhw, fractional coordinates.
        row_bboxes (ndarray): (nB, 4); ijhw, fractional coordinates.

    Returns:
        ndarray: (nA, nB); iou[a, b] = iou(col_bboxes[a], row_bboxes[b]).
    """
    ### convert: ijhw -> ijij
    col_ijij = ijhw_to_ijij(col_bboxes).astype('float16')
    row_ijij = ijhw_to_ijij(row_bboxes).astype('float16')

    ### compute intersection
    inter_lt = np.maximum(col_ijij[:, None, :2], row_ijij[:, :2])
    inter_rb = np.minimum(col_ijij[:, None, 2:], row_ijij[:, 2:])
    intersection = (inter_rb - inter_lt).clip(0).prod(2)

    ### compute union
    area_a = (col_ijij[:, 2:] - col_ijij[:, :2]).prod(1)
    area_b = (row_ijij[:, 2:] - row_ijij[:, :2]).prod(1)
    union = (area_a[:, None] + area_b) - intersection

    ### compute iou
    iou = intersection / union

    return iou.astype('float32')


def get_nms_mask(confs, bboxes, nms_iou_thresh=0.5):
    """for each set of bboxes with ious above iou_thresh, eliminates all but
    highest confidence bbox.

    Args:
        confs (ndarray): (N); confidence for each box prediction.
        bboxes (ndarray): (N, 4); unsuppressed box predictions.
        nms_iou_thresh (float): minimum iou between two boxes to be considered
            an overlap.

    Returns:
        ndarray: (N'): true where prediction should be kept.
    """
    # inds: box_inds in descending order of confidence
    inds = np.argsort(confs)[::-1]
    ious = compute_ious(bboxes, bboxes)

    inds_to_keep = set(inds)
    for ind in inds:
        if ind not in inds_to_keep:  # boxes[ind] already removed
            continue
        # { j | iou(bboxes[i], bboxes[j]) > iou_thresh }
        overlapping_bbox_inds = np.nonzero(ious[ind, :] > nms_iou_thresh)[0]
        for overlap_ind in overlapping_bbox_inds:
            if overlap_ind == ind:  # dont remove self from to_keep
                continue
            inds_to_keep.discard(overlap_ind)

    mask = np.zeros_like(confs).astype(bool)
    mask[list(inds_to_keep)] = True

    return mask
