"""PyTorch modules."""

from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class Normalizer(Module):
    """normalizes input image tensors. subclassed from nn.Module for easy
    movement of parameters to/from GPU. Normalization is required for
    pretrained backbones in model zoo.

    Args:
        mean: desired channelwise mean.
        std: desired channelwise std deviation.

    Attributes:
        mean: see Args, registered as parameter.
        std: see Args, registered as parameter.
    """

    def __init__(
        self,
        mean: Tuple[int, int, int] = (0.485, 0.456, 0.406),
        std: Tuple[int, int, int] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.mean = self._setup_parameters(mean)
        self.std = self._setup_parameters(std)

    @staticmethod
    def _setup_parameters(param: float) -> Parameter:
        """expand parameter dimensions for future broadcasting, return
        as `Parameter` for easy movement to/from GPU.
        """
        return Parameter(
            torch.as_tensor(param)[None, :, None, None], requires_grad=False
        )

    def forward(self, x: Tensor) -> Tensor:
        """input[channel] = (input[channel] - mean[channel]) / std[channel]"""
        return (x - self.mean) / self.std
