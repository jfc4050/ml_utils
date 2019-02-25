"""dataloading utilities"""

from xml.etree import ElementTree
from os import PathLike
from typing import Tuple

import numpy as np


def parse_pascal_xmlfile(
        labelpath: PathLike
)-> (Tuple[str], Tuple[Tuple[float]]):
    """extract properly formatted labels from xml file.

    Args:
        labelpath: path to xml label.

    Raises:
        FileNotFoundError: if no file at labelpath.

    Returns:
        classes: class labels.
        bboxes: bboxes labels.
    """
    root = ElementTree.parse(labelpath).getroot()

    ### get image dimensions
    size = root.find('size')
    im_h, im_w = [int(size.find(key).text) for key in ['height', 'width']]

    ### iterate through object labels
    classes, bboxes = list(), list()
    for annotation in root.findall('object'):
        classes.append(annotation.find('name').text)

        bbox = annotation.find('bndbox')
        bboxes.append([
            float(bbox.find(key).text)
            for key in ['ymin', 'xmin', 'ymax', 'xmax']
        ])

    ### convert bbox coords to ijhw, fractional coords
    ijij = np.array(bboxes) # ijij, absolute coords
    ijhw = np.concatenate(
        [
            (ijij[:, :2] + ijij[:, 2:]) / 2,  # ij
            ijij[:, 2:] - ijij[:, :2]  # hw
        ],
        axis=1
    )
    ijhw /= [im_h, im_w, im_h, im_w]  # absolute -> fractional coords

    classes = tuple(classes)
    bboxes = tuple([tuple(row) for row in ijhw])

    return classes, bboxes
