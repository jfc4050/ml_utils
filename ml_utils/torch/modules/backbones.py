import torch
from torch import nn
from torchvision.models import resnet


class Normalizer(nn.Module):
    """normalizes input image tensors. subclassed from nn.Module for easy
    movement of parameters to/from GPU. Normalization is required for
    pretrained backbones in model zoo.

    Args:
        mean ((float)): desired channelwise mean.
        std ((float)): desired channelwise std deviation.

    Attributes:
        mean (nn.Parameter): see Args, registered as parameter.
        std (nn.Parameter): see Args, registered as parameter.
    """
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.mean = self._setup_parameters(mean)
        self.std = self._setup_parameters(std)

    @staticmethod
    def _setup_parameters(param):
        """expand parameter dimensions for future broadcasting, encapsulate
        as nn.Parameter for easy movement to/from GPU.
        """
        return nn.Parameter(
            torch.Tensor(param)[None, :, None, None], requires_grad=False
        )

    def forward(self, x):
        """input[channel] = (input[channel] - mean[channel]) / std[channel]"""
        return (x - self.mean) / self.std


class ResNetBackbone(resnet.ResNet):
    """ResNet feature extraction backbone
    see: https://arxiv.org/abs/1512.03385
    should not instantiate from this class directly, see subclasses.

    Args:
        block_type (class): basic_block type.
        layers ([int]): number of blocks to use in each of four "stages".
        state_dict (dict): pretrained weights.

    Attributes:
        see superclass.
    """
    def __init__(self, block_type, layers, state_dict=None):
        """initialize layers, load and freeze weights if applicable."""
        super().__init__(block_type, layers)
        if state_dict:  # load state dict and freeze weights
            self.load_state_dict(state_dict)
            for param in self.parameters():
                param.requires_grad_(False)
        self.eval()  # batch norm -> fixed affine transform

    def forward(self, x):
        self.eval()  # ensure frozen batchnorm
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return c2, c3, c4, c5


class ResNet50Backbone(ResNetBackbone):
    """ResNet50 backbone, see superclass."""
    def __init__(self):
        state_dict = torch.load(resnet.model_urls['resnet50'])
        layers = [3, 4, 6, 3]
        super().__init__(resnet.BasicBlock, layers, state_dict)


class ResNet101Backbone(ResNetBackbone):
    """ResNet101 backbone, see superclass."""
    def __init__(self):
        state_dict = torch.load(resnet.model_urls['resnet101'])
        layers = [3, 4, 23, 3]
        super().__init__(resnet.Bottleneck, layers, state_dict)


class ResNet152Backbone(ResNetBackbone):
    """ResNet152 backbone, see superclass."""
    def __init__(self):
        state_dict = torch.load(resnet.model_urls['resnet152'])
        layers = [3, 8, 36, 3]
        super().__init__(resnet.Bottleneck, layers, state_dict)
