"""dataloading utilities"""

from xml.etree import ElementTree
from os import PathLike
from typing import Optional, NamedTuple, Tuple

import numpy as np


class PascalObjectLabel(NamedTuple):
    """pascal label for a single object"""

    class_id: str
    bbox: Tuple[float, float, float, float]
    track_id: Optional[int] = None


def parse_pascal_xmlfile(labelpath: PathLike) -> Tuple[PascalObjectLabel]:
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
    size = root.find("size")
    im_h, im_w = [int(size.find(key).text) for key in ["height", "width"]]

    ### iterate through object labels
    pascal_objects = list()
    for obj in root.findall("object"):
        if obj.find("trackid") is not None:
            track_id = int(obj.find("trackid").text)
        else:
            track_id = None
        class_id = obj.find("name").text
        bbox = obj.find("bndbox")
        ijij = np.array(
            [  # ijij, absolute coordinates
                float(bbox.find(key).text) for key in ["ymin", "xmin", "ymax", "xmax"]
            ]
        )  # ijhw, absolute coordinates
        ijhw = np.concatenate(
            [(ijij[:2] + ijij[2:]) / 2, ijij[2:] - ijij[:2]]  # ij  # hw
        )
        ijhw /= [im_h, im_w, im_h, im_w]  # ijhw, fractional coordinates
        bbox = tuple(ijhw)

        pascal_objects.append(
            PascalObjectLabel(class_id=class_id, bbox=bbox, track_id=track_id)
        )

    return tuple(pascal_objects)
