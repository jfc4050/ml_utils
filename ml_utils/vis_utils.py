import PIL.ImageDraw
import matplotlib.pyplot as plt
from bbox_utils import ijhw_to_ijij


def draw_detections(
        img,
        classes,
        boxes,
        mappings=None,
        shape='rectangle',
        cmap_key='cubehelix'
):
    """draw detections on img.
    Args:
        img (Image): PIL image to draw on.
        classes (ndarray): (N); class IDs.
        boxes (ndarray): (N, 4); bounding boxes.
        cmap_key (str): string key for matplotlib colormap.
    """
    ### conversion: ijhw, fractional -> ijij, absolute
    boxes_ijij = ijhw_to_ijij(boxes)
    im_w, im_h = img.size
    boxes_ijij *= [im_h, im_w, im_h, im_w]

    if mappings is None:
        mappings = dict()

    ### draw detections
    cmap = plt.get_cmap(cmap_key)
    draw = PIL.ImageDraw.Draw(img)
    for cls_id, (i0, j0, i1, j1) in zip(classes, boxes_ijij):
        color = tuple([int(255*x) for x in cmap(cls_id/max(classes))[:3]])

        getattr(draw, shape)([(j0, i0), (j1, i1)], outline=color)
        draw.text((j0, i0), mappings.get(cls_id, str(cls_id)), fill=color)
