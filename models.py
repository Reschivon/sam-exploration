from os import rename
from stringprep import c22_specials
from typing import Any, Optional
from torch.nn import Module, Conv2d
import torch.nn.functional as F
import resnet

from torchvision.models import resnet18
from deeplabv3 import DeepLabV3, _deeplabv3_resnet

class SteeringCommandsDQN(Module):
    def __init__(self, num_input_channels=3, num_output_channels=4):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels, num_classes=num_output_channels)

    def forward(self, x):
        return self.resnet18(x)

class DenseActionSpaceDQN(Module):
    def __init__(self, num_input_channels=3, num_output_channels=1):
        super().__init__()
        self.resnet18 = resnet.resnet18(num_input_channels=num_input_channels)
        self.conv1 = Conv2d(512, 128, kernel_size=1, stride=1)
        self.conv2 = Conv2d(128, 32, kernel_size=1, stride=1)
        self.conv3 = Conv2d(32, num_output_channels, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.resnet18.features(x)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = F.relu(self.conv2(x))
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return self.conv3(x)

# From https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py
def deeplabv3_resnet18(
    *,
    weights = None,
    progress: bool = True,
    num_classes: Optional[int] = None,
    aux_loss: Optional[bool] = None,
    num_input_channels=3,
    **kwargs: Any,
) -> DeepLabV3:
    """Constructs a DeepLabV3 model with a ResNet-18 backbone.
    .. betastatus:: segmentation module
    Reference: `Rethinking Atrous Convolution for Semantic Image Segmentation <https://arxiv.org/abs/1706.05587>`__.
    Args:
        weights (:class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.segmentation.DeepLabV3_ResNet50_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (:class:`~torchvision.models.ResNet50_Weights`, optional): The pretrained weights for the
            backbone
        **kwargs: unused
    .. autoclass:: torchvision.models.segmentation.DeepLabV3_ResNet50_Weights
        :members:
    """

    num_classes = 1

    from torchvision.models.resnet import BasicBlock
    # backbone = resnet.resnet18(num_input_channels=num_input_channels) # results in [1, 512, 32, 32]
    backbone = resnet.ResNetStock(BasicBlock, [2, 2, 2, 2], num_input_channels=num_input_channels) # results in [1, 512, 4, 4]
    model = _deeplabv3_resnet(backbone, num_classes, aux_loss)

    return model
