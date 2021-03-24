from typing import Union, Tuple, Dict, Any

from torch import nn
from torch import FloatTensor

from .resnet_modules import BasicBlock, Bottleneck
from .preact_modules import PreActBlock, PreActBottleneck

from compressors.models import BaseDistilModel


class ResNetCifar(BaseDistilModel):
    def __init__(self, depth, num_filters, block_name="BasicBlock", num_classes=10):
        super(ResNetCifar, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == "basicblock":
            assert (
                           depth - 2
                   ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == "bottleneck":
            assert (
                           depth - 2
                   ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = Bottleneck
        elif block_name.lower() == "preactblock":
            assert (
                           depth - 2
                   ) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
            block = PreActBlock
        elif block_name.lower() == "preactbottleneck":
            assert (
                           depth - 2
                   ) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
            block = PreActBottleneck
        else:
            raise ValueError("block_name shoule be Basicblock or Bottleneck")
        self.block_name = block_name
        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks - 1)))

        return nn.Sequential(*layers)

    def forward(
            self,
            x: FloatTensor,
            output_hidden_states: bool = False,
            return_dict: bool = False,
            preact: bool = False,
    ) -> Union[FloatTensor, Tuple[FloatTensor], Dict[str, FloatTensor]]:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x
        if preact and "preact" in self.block_name:
            raise Exception("PreAct blocks doesn't have activation on last layer.")
        if "preact" not in self.block_name:
            x, f1_pre = self.layer1(x)  # 32x32
        else:
            x = self.layer1(x)
        f1 = x
        if "preact" not in self.block_name:
            x, f2_pre = self.layer2(x)  # 16x16
        else:
            x = self.layer2(x)
        f2 = x
        if "preact" not in self.block_name:
            x, f3_pre = self.layer3(x)  # 8x8
        else:
            x = self.layer3(x)
        f3 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        f4 = x
        x = self.fc(x)

        if output_hidden_states:
            if preact:
                hiddens = tuple([f0, f1_pre, f2_pre, f3_pre, f4])
            else:
                hiddens = tuple([f0, f1, f2, f3, f4])
            if return_dict:
                return {"logits": x, "hidden_states": hiddens}
            return x, hiddens
        return x


def resnet_cifar_8(**kwargs) -> ResNetCifar:
    r"""
    ResNet-8 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(8, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_14(**kwargs) -> ResNetCifar:
    r"""
    ResNet-14 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(14, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_20(**kwargs) -> ResNetCifar:
    r"""
    ResNet-20 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(20, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_32(**kwargs) -> ResNetCifar:
    r"""
    ResNet-32 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(32, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_44(**kwargs) -> ResNetCifar:
    r"""
    ResNet-44 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(44, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_56(**kwargs) -> ResNetCifar:
    r"""
    ResNet-56 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(56, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_110(**kwargs) -> ResNetCifar:
    r"""
    ResNet-110 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(110, [16, 16, 32, 64], "basicblock", **kwargs)


def resnet_cifar_8x4(**kwargs) -> ResNetCifar:
    r"""
    ResNet-8x4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(8, [32, 64, 128, 256], "basicblock", **kwargs)


def resnet_cifar_32x4(**kwargs) -> ResNetCifar:
    r"""
    ResNet-32x4 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    CIFAR version.
    """
    return ResNetCifar(32, [32, 64, 128, 256], "basicblock", **kwargs)


def preact_resnet_cifar_8(**kwargs) -> ResNetCifar:
    return ResNetCifar(8, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_14(**kwargs) -> ResNetCifar:
    return ResNetCifar(14, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_20(**kwargs) -> ResNetCifar:
    return ResNetCifar(20, [16, 16, 32, 64], "preaactblock", **kwargs)


def preact_resnet_cifar_32(**kwargs: Any) -> ResNetCifar:
    return ResNetCifar(32, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_44(**kwargs) -> ResNetCifar:
    return ResNetCifar(44, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_56(**kwargs) -> ResNetCifar:
    return ResNetCifar(56, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_110(**kwargs) -> ResNetCifar:
    return ResNetCifar(110, [16, 16, 32, 64], "preactblock", **kwargs)


def preact_resnet_cifar_8x4(**kwargs) -> ResNetCifar:
    return ResNetCifar(8, [32, 64, 128, 256], "preactblock", **kwargs)


def preact_resnet_cifar_32x4(**kwargs) -> ResNetCifar:
    return ResNetCifar(32, [32, 64, 128, 256], "preactblock", **kwargs)
