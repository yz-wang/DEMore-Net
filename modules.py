import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
import scipy.stats as st


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    out_filter = np.repeat(out_filter, channels, axis=2)
    out_filter = np.repeat(out_filter, channels, axis=3)
    return out_filter


class Blur(nn.Module):
    def __init__(self, nc):
        super(Blur, self).__init__()
        kernel = gauss_kernel(21, 3, nc)
        kernel = torch.from_numpy(kernel).permute(3, 2, 0, 1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x = F.conv2d(x, self.weight, stride=1, padding=10)
        return x


class DGNL(nn.Module):
    def __init__(self, in_channels):
        super(DGNL, self).__init__()

        self.eps = 1e-6
        self.sigma_pow2 = 100

        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x, depth_map):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)

        ### appearance relation map
        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)

        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        Ra = F.softmax(torch.bmm(theta, phi), 2)

        ### depth relation map
        depth1 = F.upsample(depth_map, size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(n, 1,
                                                                                                                int(
                                                                                                                    h / 4) * int(
                                                                                                                    w / 4)).transpose(
            1, 2)
        depth2 = F.upsample(depth_map, size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(n, 1,
                                                                                                                int(
                                                                                                                    h / 8) * int(
                                                                                                                    w / 8))

        # n, (h / 4) * (w / 4), (h / 8) * (w / 8)
        depth1_expand = depth1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        depth2_expand = depth2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))

        Rd = torch.min(depth1_expand / (depth2_expand + self.eps), depth2_expand / (depth1_expand + self.eps))

        # normalization: depth relation map [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        # Rd = Rd / (torch.sum(Rd, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)

        Rd = F.softmax(Rd, 2)

        # ### position relation map
        # position_h = torch.Tensor(range(h)).cuda().view(h, 1).expand(h, w)
        # position_w = torch.Tensor(range(w)).cuda().view(1, w).expand(h, w)
        #
        # position_h1 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1,2)
        # position_h2 = F.interpolate(position_h.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_h1_expand = position_h1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_h2_expand = position_h2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # h_distance = (position_h1_expand - position_h2_expand).pow(2)
        #
        # position_w1 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 4), int(w / 4)], mode='bilinear', align_corners=True).view(1, 1, int(h / 4) * int(w / 4)).transpose(1, 2)
        # position_w2 = F.interpolate(position_w.unsqueeze(0).unsqueeze(0), size=[int(h / 8), int(w / 8)], mode='bilinear', align_corners=True).view(1, 1, int(h / 8) * int(w / 8))
        # position_w1_expand = position_w1.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # position_w2_expand = position_w2.expand(n, int(h / 4) * int(w / 4), int(h / 8) * int(w / 8))
        # w_distance = (position_w1_expand - position_w2_expand).pow(2)
        #
        # Rp = 1 / (2 * 3.14159265 * self.sigma_pow2) * torch.exp(-0.5 * (h_distance / self.sigma_pow2 + w_distance / self.sigma_pow2))
        #
        # Rp = Rp / (torch.sum(Rp, 2).view(n, int(h / 4) * int(w / 4), 1) + self.eps)

        ### overal relation map
        # S = F.softmax(Ra * Rd * Rp, 2)

        S = F.softmax(Ra * Rd, 2)

        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(S, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)


def get_1d_dct(i, freq, L):
    result = math.cos(math.pi * freq * (i + 0.5) / L) / math.sqrt(L)
    if freq == 0:
        return result
    else:
        return result * math.sqrt(2)


def get_dct_weights(width, height, channel, fidx_u=[0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 2, 3], fidx_v=[0, 1, 0, 5,
                                                                                                             2, 0, 2, 0,
                                                                                                             0, 6, 0, 4,
                                                                                                             6, 3, 2,
                                                                                                             5]):
    # width : width of input
    # height : height of input
    # channel : channel of input
    # fidx_u : horizontal indices of selected fequency
    # according to the paper, should be [0,0,6,0,0,1,1,4,5,1,3,0,0,0,2,3]
    # fidx_v : vertical indices of selected fequency
    # according to the paper, should be [0,1,0,5,2,0,2,0,0,6,0,4,6,3,2,5]
    # [0,0],[0,1],[6,0],[0,5],[0,2],[1,0],[1,2],[4,0],
    # [5,0],[1,6],[3,0],[0,4],[0,6],[0,3],[2,2],[3,5],
    scale_ratio = width // 7
    fidx_u = [u * scale_ratio for u in fidx_u]
    fidx_v = [v * scale_ratio for v in fidx_v]
    dct_weights = torch.zeros(1, channel, width, height)
    c_part = channel // len(fidx_u)
    # split channel for multi-spectal attention
    for i, (u_x, v_y) in enumerate(zip(fidx_u, fidx_v)):
        for t_x in range(width):
            for t_y in range(height):
                dct_weights[:, i * c_part: (i + 1) * c_part, t_x, t_y] \
                    = get_1d_dct(t_x, u_x, width) * get_1d_dct(t_y, v_y, height)
    # Eq. 7 in our paper
    return dct_weights


class FcaLayer(nn.Module):
    def __init__(self, channel, reduction, width, height):
        super(FcaLayer, self).__init__()
        self.width = width
        self.height = height
        self.register_buffer('pre_computed_dct_weights', get_dct_weights(self.width, self.height, channel))
        # self.register_parameter('pre_computed_dct_weights',torch.nn.Parameter(get_dct_weights(width,height,channel)))
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, (self.height, self.width))
        s= y * self.pre_computed_dct_weights
        y = torch.sum(y * self.pre_computed_dct_weights, dim=(2, 3))
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DepthWiseDilatedResidualBlock(nn.Module):
    def __init__(self, reduced_channels, channels, dilation):
        super(DepthWiseDilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(

            # pw
            nn.Conv2d(channels, channels * 2, 1, 1, 0, 1, bias=False),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(channels * 2, channels * 2, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
                      bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(channels * 2, channels, 1, 1, 0, 1, 1, bias=False)
        )

        self.conv1 = nn.Sequential(
            # pw
            # nn.Conv2d(channels, channels * 2, 1, 1, 0, 1, bias=False),
            # nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, groups=channels,
                      bias=False),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(channels, channels, 1, 1, 0, 1, 1, bias=False)
        )

    def forward(self, x):
        res = self.conv1(self.conv0(x))
        return res + x


class ConvBlock(nn.Module):
    """ implement conv+ReLU two times """

    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        conv_relu = []
        conv_relu.append(nn.Conv2d(in_channels=in_channels, out_channels=middle_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        conv_relu.append(nn.Conv2d(in_channels=middle_channels, out_channels=out_channels,
                                   kernel_size=3, padding=1, stride=1))
        conv_relu.append(nn.ReLU())
        self.conv_ReLU = nn.Sequential(*conv_relu)

    def forward(self, x):
        out = self.conv_ReLU(x)
        return out


class U_Net(nn.Module):
    def __init__(self):
        super().__init__()

        # 首先定义左半部分网络
        # left_conv_1 表示连续的两个（卷积+激活）
        # 随后进行最大池化
        self.left_conv_1 = ConvBlock(in_channels=64, middle_channels=64, out_channels=64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_2 = ConvBlock(in_channels=64, middle_channels=128, out_channels=128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_3 = ConvBlock(in_channels=128, middle_channels=256, out_channels=256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_4 = ConvBlock(in_channels=256, middle_channels=512, out_channels=512)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.left_conv_5 = ConvBlock(in_channels=512, middle_channels=1024, out_channels=1024)

        # 定义右半部分网络
        self.deconv_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.right_conv_1 = ConvBlock(in_channels=1024, middle_channels=512, out_channels=512)

        self.deconv_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_2 = ConvBlock(in_channels=512, middle_channels=256, out_channels=256)

        self.deconv_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, stride=2,
                                           output_padding=1)
        self.right_conv_3 = ConvBlock(in_channels=256, middle_channels=128, out_channels=128)

        self.deconv_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, output_padding=1,
                                           padding=1)
        self.right_conv_4 = ConvBlock(in_channels=128, middle_channels=64, out_channels=64)
        # 最后是1x1的卷积，用于将通道数化为3
        self.right_conv_5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_5)
        # 特征拼接
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)

        return out


class Constage(nn.Module):
    def __init__(self, k_size, stride, out_dims, input_dims):
        super(Constage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dims, out_dims, k_size, stride, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        output = self.layers(x)

        return output


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels, dilation):
        super(DilatedResidualBlock, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation), nn.ReLU()
        )
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        return x + conv1


class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1, downsample=None):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=o_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(nn.Module):
    # 224*224
    def __init__(self, block, num_layer, n_classes=1000, input_channels=3):
        super(Resnet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_layer[0])
        self.layer2 = self._make_layer(block, 128, num_layer[1], 2)
        self.layer3 = self._make_layer(block, 256, num_layer[2], 2)
        self.layer4 = self._make_layer(block, 512, num_layer[3], 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(block.expansion * 512, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_block):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SpatialRNN(nn.Module):
    """
    SpatialRNN model for one direction only
    """

    def __init__(self, alpha=1.0, channel_num=1, direction="right"):
        super(SpatialRNN, self).__init__()
        self.alpha = nn.Parameter(torch.Tensor([alpha] * channel_num))
        self.direction = direction

    def __getitem__(self, item):
        return self.alpha[item]

    def __len__(self):
        return len(self.alpha)

    def forward(self, x):
        """
        :param x: (N,C,H,W)
        :return:
        """
        height = x.size(2)
        weight = x.size(3)
        x_out = []

        # from left to right
        if self.direction == "right":
            x_out = [x[:, :, :, 0].clamp(min=0)]

            for i in range(1, weight):
                temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, i]).clamp(min=0)
                x_out.append(temp)  # a list of tensor

            return torch.stack(x_out, 3)  # merge into one tensor

        # from right to left
        elif self.direction == "left":
            x_out = [x[:, :, :, -1].clamp(min=0)]

            for i in range(1, weight):
                temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, :, -i - 1]).clamp(min=0)
                x_out.append(temp)

            x_out.reverse()
            return torch.stack(x_out, 3)

        # from up to down
        elif self.direction == "down":
            x_out = [x[:, :, 0, :].clamp(min=0)]

            for i in range(1, height):
                temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, i, :]).clamp(min=0)
                x_out.append(temp)

            return torch.stack(x_out, 2)

        # from down to up
        elif self.direction == "up":
            x_out = [x[:, :, -1, :].clamp(min=0)]

            for i in range(1, height):
                temp = (self.alpha.unsqueeze(1) * x_out[i - 1] + x[:, :, -i - 1, :]).clamp(min=0)
                x_out.append(temp)

            x_out.reverse()
            return torch.stack(x_out, 2)

        else:
            print("Invalid direction in SpatialRNN!")
            return KeyError


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class NLB(nn.Module):
    def __init__(self, in_channels):
        super(NLB, self).__init__()
        self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
        self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)

        self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
        self.down.weight.data.fill_(1. / 16)

        self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.size()
        x_down = self.down(x)

        # [n, (h / 4) * (w / 4), c / 2]
        theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, c / 2, (h / 8) * (w / 8)]
        phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
        # [n, (h / 8) * (w / 8), c / 2]
        g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
        # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
        f = F.softmax(torch.bmm(theta, phi), 2)
        # [n, c / 2, h / 4, w / 4]
        y = torch.bmm(f, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))

        return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)

# class DGNLB(nn.Module):
#     def __init__(self, in_channels):
#         super(DGNLB, self).__init__()
#
#         self.roll = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#         self.ita = nn.Conv2d(1, int(in_channels / 2), kernel_size=1)
#
#         self.theta = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.phi = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#         self.g = nn.Conv2d(in_channels, int(in_channels / 2), kernel_size=1)
#
#         self.down = nn.Conv2d(in_channels, in_channels, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         self.down.weight.data.fill_(1. / 16)
#
#         #self.down_depth = nn.Conv2d(1, 1, kernel_size=4, stride=4, groups=in_channels, bias=False)
#         #self.down_depth.weight.data.fill_(1. / 16)
#
#         self.z = nn.Conv2d(int(in_channels / 2), in_channels, kernel_size=1)
#
#     def forward(self, x, depth):
#         n, c, h, w = x.size()
#         x_down = self.down(x)
#
#         depth_down = F.avg_pool2d(depth, kernel_size=(4,4))
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         #roll = self.roll(depth_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 4) * (w / 4)]
#         #ita = self.ita(depth_down).view(n, int(c / 2), -1)
#         # [n, (h / 4) * (w / 4), (h / 4) * (w / 4)]
#
#         depth_correlation = F.softmax(torch.bmm(depth_down.view(n, 1, -1).transpose(1, 2), depth_down.view(n, 1, -1)), 2)
#
#
#         # [n, (h / 4) * (w / 4), c / 2]
#         theta = self.theta(x_down).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, c / 2, (h / 8) * (w / 8)]
#         phi = F.max_pool2d(self.phi(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1)
#         # [n, (h / 8) * (w / 8), c / 2]
#         g = F.max_pool2d(self.g(x_down), kernel_size=2, stride=2).view(n, int(c / 2), -1).transpose(1, 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         f_correlation = F.softmax(torch.bmm(theta, phi), 2)
#         # [n, (h / 4) * (w / 4), (h / 8) * (w / 8)]
#         final_correlation = F.softmax(torch.bmm(depth_correlation, f_correlation), 2)
#
#         # [n, c / 2, h / 4, w / 4]
#         y = torch.bmm(final_correlation, g).transpose(1, 2).contiguous().view(n, int(c / 2), int(h / 4), int(w / 4))
#
#         return x + F.upsample(self.z(y), size=x.size()[2:], mode='bilinear', align_corners=True)
