# From: https://github.com/pytorch/vision/blob/main/torchvision/models/segmentation/deeplabv3.py

from functools import partial
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet
from torchvision.models.segmentation._utils import _SimpleSegmentationModel
from torchvision.models.segmentation.fcn import FCNHead


__all__ = [
    "DeepLabV3",
    "DeepLabV3_ResNet50_Weights",
    "DeepLabV3_ResNet101_Weights",
    "DeepLabV3_MobileNet_V3_Large_Weights",
    "deeplabv3_mobilenet_v3_large",
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.

    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """

    pass


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    aux: Optional[bool],
) -> DeepLabV3:
    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = FCNHead(1024, num_classes) if aux else None
    classifier = DeepLabHead(2048, num_classes)
    return DeepLabV3(backbone, classifier, aux_classifier)


# def _deeplabv3_mobilenetv3(
#     backbone: MobileNetV3,
#     num_classes: int,
#     aux: Optional[bool],
# ) -> DeepLabV3:
#     backbone = backbone.features
#     # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
#     # The first and last blocks are always included because they are the C0 (conv1) and Cn.
#     stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
#     out_pos = stage_indices[-1]  # use C5 which has output_stride = 16
#     out_inplanes = backbone[out_pos].out_channels
#     aux_pos = stage_indices[-4]  # use C2 here which has output_stride = 8
#     aux_inplanes = backbone[aux_pos].out_channels
#     return_layers = {str(out_pos): "out"}
#     if aux:
#         return_layers[str(aux_pos)] = "aux"
#     backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

#     aux_classifier = FCNHead(aux_inplanes, num_classes) if aux else None
#     classifier = DeepLabHead(out_inplanes, num_classes)
#     return DeepLabV3(backbone, classifier, aux_classifier)
