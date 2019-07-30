"""data augmentation layer classes"""

import abc
import random

import numpy as np
from torchvision import transforms
import PIL

from ..boundingboxes import ijhw_to_ijij


class AugmentationLayer(abc.ABC):
    """performs random transformation of image and bboxes."""

    @abc.abstractmethod
    def __call__(self, img, bboxes):
        """perform augmentation on image and adjust bboxes accordingly.

        Args:
            img (Image): input image.
            bboxes (ndarray): expected in ijhw, fractional coordinates.

        Returns:
            Image: augmented image.
            ndarray: adjusted bounding boxes.
        """
        raise NotImplementedError


class AugmentationPipeline(AugmentationLayer):
    """used to express multiple composed transformations as a single
    transformation layer.

    Args:
        *aug_layers ([AugmentationLayer]): list of transforms to apply.

    Attributes:
        aug_layers ([AugmentationLayer]): see Args.
    """

    def __init__(self, *aug_layers):
        self.aug_layers = aug_layers

    def __call__(self, img, bboxes):
        """see superclass"""
        for layer in self.aug_layers:
            img, bboxes = layer(img, bboxes)
        return img, bboxes


class Jitter(AugmentationLayer):
    """randomly change image color properties.

    Args:
        brightness (float): brightness jitter.
        contrast (float): contrast jitter.
        saturation (float): saturation jitter.
        hue (float): hue jitter.

    Attributes:
        jitter (Transform): unwrapped transformation layer.
    """

    def __init__(self, brightness=0.5, contrast=0.5, saturation=2.0, hue=0.05):
        self.jitter = transforms.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, bboxes):
        """see superclass."""
        img = self.jitter(img)
        return img, bboxes


class GrayScale(AugmentationLayer):
    """random color -> grayscale conversion.

    Args:
        p (float): probability that image will be converted to grayscale.

    Attributes:
        gs (Transform): unwrapped transformation layer.
    """

    def __init__(self, p=0.2):
        self.gs = transforms.RandomGrayscale(p)

    def __call__(self, img, bboxes):
        """see superclass."""
        img = self.gs(img)
        return img, bboxes


class RandHFlip(AugmentationLayer):
    """random horizontal flipping

    Args:
        p (float): probability that image will be horizontally flipped.

    Attributes:
        p (float): see Args.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        """see superclass."""
        if random.random() < self.p:
            img = transforms.functional.hflip(img)
            bboxes = bboxes.copy()
            bboxes[:, 1] = 1 - bboxes[:, 1]  # invert j coordinates

        return img, bboxes


class RandCrop(AugmentationLayer):
    """take a random crop out of image without cropping out any positives."""

    def __call__(self, img, bboxes):
        """see superclass."""
        ### randomly select crop points (fractional coords)
        bboxes_ijij = ijhw_to_ijij(bboxes)
        crop_i0 = np.random.uniform(0, min(bboxes_ijij[:, 0]))
        crop_j0 = np.random.uniform(0, min(bboxes_ijij[:, 1]))
        crop_i1 = np.random.uniform(max(bboxes_ijij[:, 2]), 1)
        crop_j1 = np.random.uniform(max(bboxes_ijij[:, 3]), 1)
        crop_ijij = np.array([crop_i0, crop_j0, crop_i1, crop_j1])

        ### convert crop points to absolute coordinates
        full_w, full_h = img.size
        crop_ijij_abs = (crop_ijij * [full_h, full_w, full_h, full_w]).astype(int)
        crop_i0_abs, crop_j0_abs, crop_i1_abs, crop_j1_abs = crop_ijij_abs

        ### crop image
        img_arr = np.array(img)
        img_arr = img_arr[crop_i0_abs:crop_i1_abs, crop_j0_abs:crop_j1_abs]
        img = PIL.Image.fromarray(img_arr)

        ### adjust bounding boxes in absolute coordinates,
        ### then convert back to fractional (for new image dimensions)
        bboxes_abs = bboxes * [full_h, full_w, full_h, full_w]
        bboxes_abs[:, :2] -= [crop_i0_abs, crop_j0_abs]
        crop_w, crop_h = img.size
        bboxes = bboxes_abs / [crop_h, crop_w, crop_h, crop_w]

        return img, bboxes
