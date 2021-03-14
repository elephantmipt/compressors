from torch import nn
from torch import FloatTensor

from .resnet_modules import BasicBlock, Bottleneck

from compressors.models import BaseDistilModel


class ResNetCifar(BaseDistilModel):

    def __init__(self, depth, num_filters, block_name='BasicBlock', num_classes=10):
        super(ResNetCifar, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')

        self.inplanes = num_filters[0]
        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = list([])
        layers.append(block(self.inplanes, planes, stride, downsample, is_last=(blocks == 1)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, is_last=(i == blocks-1)))

        return nn.Sequential(*layers)

    def get_feat_modules(self):
        feat_m = nn.ModuleList([])
        feat_m.append(self.conv1)
        feat_m.append(self.bn1)
        feat_m.append(self.relu)
        feat_m.append(self.layer1)
        feat_m.append(self.layer2)
        feat_m.append(self.layer3)
        return feat_m

    def get_bn_before_relu(self):
        if isinstance(self.layer1[0], Bottleneck):
            bn1 = self.layer1[-1].bn3
            bn2 = self.layer2[-1].bn3
            bn3 = self.layer3[-1].bn3
        elif isinstance(self.layer1[0], BasicBlock):
            bn1 = self.layer1[-1].bn2
            bn2 = self.layer2[-1].bn2
            bn3 = self.layer3[-1].bn2
        else:
            raise NotImplementedError('ResNet unknown block error !!!')

        return [bn1, bn2, bn3]

    def forward(
            self, x: FloatTensor, output_hidden_states: bool = False, return_dict: bool = False, preact: bool = False
    ) -> FloatTensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32
        f0 = x

        x, f1_pre = self.layer1(x)  # 32x32
        f1 = x
        x, f2_pre = self.layer2(x)  # 16x16
        f2 = x
        x, f3_pre = self.layer3(x)  # 8x8
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


def resnet_cifar_8(**kwargs):
    return ResNetCifar(8, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_14(**kwargs):
    return ResNetCifar(14, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_20(**kwargs):
    return ResNetCifar(20, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_32(**kwargs):
    return ResNetCifar(32, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_44(**kwargs):
    return ResNetCifar(44, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_56(**kwargs):
    return ResNetCifar(56, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_110(**kwargs):
    return ResNetCifar(110, [16, 16, 32, 64], 'basicblock', **kwargs)


def resnet_cifar_8x4(**kwargs):
    return ResNetCifar(8, [32, 64, 128, 256], 'basicblock', **kwargs)


def resnet_cifar_32x4(**kwargs):
    return ResNetCifar(32, [32, 64, 128, 256], 'basicblock', **kwargs)
