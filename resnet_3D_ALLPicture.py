
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F




class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.cs = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_out, max_out], dim=1)
        a = self.cs(a)
        return x*a



def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    #         model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    #         def get_inplanes():
    #               return [64, 128, 256, 512]
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=1, #
                 conv1_t_size=7,
                 conv1_t_stride=2,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400):
        super().__init__()
                                                    #  [64, 128, 256, 512]
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.Spatial_attention = SpatialAttention()

        # self.conv1 = nn.Conv3d(n_input_channels,
        #                        self.in_planes,
        #                        kernel_size=(conv1_t_size, 7, 7),
        #                        # kernel_size=(1, 1, 1),
        #
        #                        stride=(conv1_t_stride, 2, 2),
        #                        padding=(conv1_t_size // 2, 3, 3),
        #                        bias=False)
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=3,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        # self.Spatial_attention = SpatialAttention()

        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        #         return nn.Sequential(*layers)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):  # blocks = 2
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        print(x.shape)  # torch.Size([16, 1, 62, 62, 62])
        x = self.Spatial_attention(x)
        print(x.shape)  # torch.Size([16, 64, 16, 16, 16])
        x = self.conv1(x)
        print(x.shape)  # torch.Size([16, 64, 16, 16, 16])

        x = self.bn1(x)
        x = self.relu(x)  # torch.Size([16, 64, 16, 16, 16])
        if not self.no_max_pool:
            x = self.maxpool(x)

        # x = self.Spatial_attention(x)

        x = self.layer1(x)
        print(x.shape)  # torch.Size([32, 64, 25, 13, 13])
        x = self.layer2(x)
        print(x.shape)  # torch.Size([32, 128, 13, 7, 7])
        x = self.layer3(x)
        print(x.shape)  # torch.Size([32, 256, 7, 4, 4])
        x = self.layer4(x)
        print(x.shape)  # torch.Size([32, 512, 4, 2, 2])

        x = self.avgpool(x)
        print(x.shape)  # torch.Size([32, 512, 1, 1, 1])

        x = x.view(x.size(0), -1)
        # print(x.shape)  # torch.Size([32, 512])
        x = self.fc(x)
        # print(x.shape)  # torch.Size([32, 400])
        # print(1111)
        #
        # fc2 = nn.Linear(400, 2)
        # softmax = nn.Softmax(dim=1)
        # x = fc2(x)
        # x = softmax(x)
        # print(x.shape)

        return x


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model

if __name__ == "__main__":
    test = generate_model(18)
    input_x = torch.randn((16, 1, 181, 217, 181))
    output_x = test(input_x)
    print(output_x.shape)
    # print(test)
