import torch
from torch import nn
from collections import OrderedDict
from ExternalAttention import External_attention


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DenseLayer, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            External_attention(out_channels)
        )
    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        return torch.cat([x, conv1], 1)


class DenseBlock(nn.Module):
    def __init__(self):
        super(DenseBlock, self).__init__()
        self.block = []
        self.block.append(DenseLayer(64, 64, 1)),
        self.block.append(DenseLayer(128, 64, 1)),
        self.block.append(DenseLayer(192, 64, 2)),
        self.block.append(DenseLayer(256, 64, 2)),
        self.block.append(DenseLayer(320, 64, 4)),
        self.block.append(DenseLayer(384, 64, 8)),
        self.block.append(DenseLayer(448, 64, 4)),
        self.block.append(DenseLayer(512, 64, 2)),
        self.block.append(DenseLayer(576, 64, 2)),
        self.block.append(DenseLayer(640, 64, 1)),
        # for i in range(num_layers - 1):
        #     p = [1, 1, 2, 2, 4, 8, 4, 2, 2, 1, 1]
        #     self.block.append(DenseLayer(growth_rate * (i + 1), growth_rate, p[i]))

        self.block = nn.Sequential(*self.block)

    def forward(self, x):
        return torch.cat([x, self.block(x)], 1)


class DenseRR(nn.Module):
    "Denserr model"
    def __init__(self):

        super(DenseRR, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)),
            ('relu00', nn.ReLU(inplace=True)),
            ('conv001', nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # DenseBlock
        # high level features
        self.dense_blocks = []
        self.dense_blocks.append(DenseBlock())
        # self.dense_blocks.append(DenseLayer(64, 64, 1)),
        # self.dense_blocks.append(DenseLayer(128, 64, 1)),
        # self.dense_blocks.append(DenseLayer(192, 64, 2)),
        # self.dense_blocks.append(DenseLayer(256, 64, 2)),
        # self.dense_blocks.append(DenseLayer(320, 64, 4)),
        # self.dense_blocks.append(DenseLayer(384, 64, 8)),
        # self.dense_blocks.append(DenseLayer(448, 64, 4)),
        # self.dense_blocks.append(DenseLayer(512, 64, 2)),
        # self.dense_blocks.append(DenseLayer(576, 64, 2)),
        # self.dense_blocks.append(DenseLayer(640, 64, 1)),
        # self.dense_blocks.append(DenseLayer(704, 64, 1)),
        self.dense_blocks = nn.Sequential(*self.dense_blocks)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = self.dense_blocks(features)
        # out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        return out


# class _DDRB(nn.Sequential):
#     """ ConvNet block for building DenseDRB. """
#     def __init__(self, channels, dilation, drop_out):
#         super(_DDRB, self).__init__()
#
#         # self.add_module('norm01', bn(input_num)),
#         # self.add_module('relu01', nn.ReLU(inplace=True)),
#         self.add_module('conv01', nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)),
#         self.add_module('relu01', nn.ReLU()),
#         self.add_module('conv02', nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)),
#
#         self.drop_rate = drop_out
#
#     def forward(self, _input):
#         feature = super(_DDRB, self).forward(_input)
#
#         if self.drop_rate > 0:
#             feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
#         return feature


