from typing import Union, Tuple, List

from torch import Tensor
from torch import nn
from torch.nn import functional as F

from modules.resnet import PreActBlock, PreActBottleneck

from src.models import BaseDistilModel


class PreActResNet(BaseDistilModel):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(
        self, x: Tensor, output_hidden_states: bool = False, return_dict: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, List[Tensor]]]:
        """
        Forward pass.
        Args:
            x: input.
            output_hidden_states: hidden states from model.
            return_dict: wheather to output dict or tuple.
        Returns:
            logits or tuple of logits and list of hiddens.
        """
        out = self.conv1(x)
        layers = [f"layer{i}" for i in range(1, 5)]
        if output_hidden_states:
            hiddens = []
        for l_name in layers:
            layer = getattr(self, l_name)
            out = layer(out)
            if output_hidden_states:
                hiddens.append(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if output_hidden_states:
            hiddens.append(out)
        out = self.linear(out)
        if return_dict:
            out = {"logits": out}
            if output_hidden_states:
                out["hidden_states"] = tuple(hiddens)
        else:
            if output_hidden_states:
                out = (out, tuple(hiddens))
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])


__all__ = [
    "PreActResNet18",
    "PreActResNet34",
    "PreActResNet50",
    "PreActResNet101",
    "PreActResNet152",
]
