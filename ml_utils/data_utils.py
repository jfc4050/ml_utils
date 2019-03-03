"""dataloading utilities"""

from xml.etree import ElementTree
from os import PathLike
from typing import NamedTuple, Tuple

import numpy as np


class PascalObjectLabel(NamedTuple):
    """pascal label for a single object"""
    track_id: int
    class_id: str
    bbox: Tuple[float]


def parse_pascal_xmlfile(
        labelpath: PathLike
)-> Tuple[PascalObjectLabel]:
    """extract properly formatted labels from xml file.

    Args:
        labelpath: path to xml label.

    Raises:
        FileNotFoundError: if no file at labelpath.

    Returns:
        pascal_objects: tuple of pascal objects
    """
    root = ElementTree.parse(labelpath).getroot()

    ### get image dimensions
    size = root.find('size')
    im_h, im_w = [int(size.find(key).text) for key in ['height', 'width']]

    ### iterate through object labels
    pascal_objects = list()
    for obj in root.findall('object'):
        track_id = int(obj.find('trackid').text)
        class_id = obj.find('name').text
        bbox = obj.find('bndbox')
        ijij = np.array([  # ijij, absolute coordinates
            float(bbox.find(key).text)
            for key in ['ymin', 'xmin', 'ymax', 'xmax']
        ])  # ijhw, absolute coordinates
        ijhw = np.concatenate([
            (ijij[:2] + ijij[2:]) / 2,  # ij
            ijij[2:] - ijij[:2]  # hw
        ])
        ijhw /= [im_h, im_w, im_h, im_w]  # ijhw, fractional coordinates
        bbox = tuple(ijhw)

        pascal_objects.append(PascalObjectLabel(track_id, class_id, bbox))

    return tuple(pascal_objects)
