"""visualization utilities"""

from typing import Dict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from .bbox_utils import ijhw_to_ijij


def draw_detections(
    img: Image,
    confs: np.ndarray,
    classes: np.ndarray,
    boxes: np.ndarray,
    mappings: Dict[int, str] = None,
    cmap_key: str = "cubehelix",
) -> None:
    """draw detections on image.

    Args:
        img (Image): PIL image to draw on.
        confs (ndarray): (|D|,) prediction confidences.
        classes (ndarray): (|D|,) class IDs.
        boxes (ndarray): (|D|, 4) bounding boxes.
        mappings (dict): class id (int) -> class name (str).
        cmap_key (str): string key for matplotlib colormap.
    """
    # conversion: ijhw, fractional -> ijij, absolute
    im_w, im_h = img.size
    boxes_ijij = ijhw_to_ijij(boxes) * [im_h, im_w, im_h, im_w]

    mappings = mappings or dict()
    confs = confs if confs is not None else np.ones(len(classes), dtype=float)

    ### draw detections
    cmap = plt.get_cmap(cmap_key)
    draw = ImageDraw.Draw(img)
    for conf, cls_id, (i0, j0, i1, j1) in zip(confs, classes, boxes_ijij):
        color = tuple([int(255 * x) for x in cmap(cls_id / max(classes))[:3]])

        draw.rectangle([(j0, i0), (j1, i1)], outline=color)
        draw.text(
            (j0, i0), f"{mappings.get(cls_id, str(cls_id))}-{conf:.2f}", fill=color
        )
