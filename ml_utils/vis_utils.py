"""visualization utilities."""

from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .boundingboxes import ijhw_to_ijij


def draw_detections(
    img: Image, bboxes: np.ndarray, labels: Sequence[str] = None
) -> None:
    """draw detections on image.

    Args:
        img: PIL image to draw on.
        bboxes: (|D|, 4) bounding boxes.
        labels: text label for each bounding box.
    """
    im_w, im_h = img.size
    boxes_ijij = ijhw_to_ijij(bboxes) * [im_h, im_w, im_h, im_w]
    labels = labels or [""] * len(bboxes)

    draw = ImageDraw.Draw(img)
    for label, (i0, j0, i1, j1) in zip(labels, boxes_ijij):
        draw.rectangle([(j0, i0), (j1, i1)])
        draw.text((j0, i0), label)
