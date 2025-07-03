# coding=utf-8
from SN_YL import SNConv2d
import torch.nn as nn


class DiscriminativeNet(nn.Module):
    def __init__(self, input_nc=3, ndf=64):
        super(DiscriminativeNet, self).__init__()

        self.model = nn.Sequential(
            # input is 3 x 256 x 256
            #SNConv2d()
            SNConv2d(input_nc, ndf, 6, 2, 2),
            # nn.InstanceNorm2d(ndf),
            nn.ReLU(inplace=True),
            SNConv2d(ndf, ndf * 2, 6, 2, 2),
            # nn.InstanceNorm2d(ndf * 2),
            nn.ReLU(inplace=True),
            # state size
            SNConv2d(ndf * 2, 256, 3, 1, 1),
            # nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            SNConv2d(256, 512, 3, 1, 1),
            # nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            # state size 输出1 * 64 * 64
            SNConv2d(512, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        #self.snlinear = nn.Sequential(SNLinear(ndf * 4 * 4 * 4, 1),
        #                              nn.Sigmoid())

    def forward(self, x):
        output = self.model(x)
        #output = output.view(output.size(0), -1)
        #output = self.snlinear(output)
        return output


