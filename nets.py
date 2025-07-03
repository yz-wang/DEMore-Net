import torch
import torch.nn.functional as F
from torch import nn
from modules import DGNL
# from DSC import DSC_Module
from torch.autograd import Variable
from torchvision import models
from torch import optim
from Denserr import DenseRR
from ExternalAttention import External_attention


class IBNorm(nn.Module):
    """ Combine Instance Norm and Batch Norm into One Layer
    """

    def __init__(self, in_channels):
        super(IBNorm, self).__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels

        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)  # Half Batch & Instance

    def forward(self, x):
        bn_x = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x = self.inorm(x[:, self.bnorm_channels:, ...].contiguous())

        return torch.cat((bn_x, in_x), 1)


class GenerativeNetwork(nn.Module):
    def __init__(self, num_features=64):
        super(GenerativeNetwork, self).__init__()
        self.mean = torch.zeros(1, 3, 1, 1)
        self.std = torch.zeros(1, 3, 1, 1)
        self.mean[0, 0, 0, 0] = 0.485
        self.mean[0, 1, 0, 0] = 0.456
        self.mean[0, 2, 0, 0] = 0.406
        self.std[0, 0, 0, 0] = 0.229
        self.std[0, 1, 0, 0] = 0.224
        self.std[0, 2, 0, 0] = 0.225

        self.mean = nn.Parameter(self.mean)
        self.std = nn.Parameter(self.std)
        self.mean.requires_grad = False
        self.std.requires_grad = False

        ############################################ Depth prediction network
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            Residual_Block(32, 32),
            IBNorm(in_channels=32),
            nn.SELU(inplace=True),
            # External_attention(32)
            SCBottleneck(32, 32)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            Residual_Block(64, 64),
            IBNorm(in_channels=64),
            nn.SELU(inplace=True),
            # External_attention(64)
            SCBottleneck(64, 64)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            Residual_Block(128, 128),
            IBNorm(in_channels=128),
            nn.SELU(inplace=True),
            # External_attention(128)
            SCBottleneck(128, 128)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            Residual_Block(256, 256),
            IBNorm(in_channels=256),
            nn.SELU(inplace=True),
            # External_attention(256)
            SCBottleneck(256, 256)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            Residual_Block(256, 256),
            IBNorm(in_channels=256),
            nn.SELU(inplace=True),
            # External_attention(256),
            SCBottleneck(256, 256)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=4, dilation=4),
            Residual_Block(256, 256),
            IBNorm(in_channels=256),
            nn.SELU(inplace=True),
            # External_attention(256)
            SCBottleneck(256, 256)
        )
        # 6,7 layer are same
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=2, dilation=2),
            Residual_Block(256, 256),
            IBNorm(in_channels=256),
            nn.SELU(inplace=True),
            # External_attention(256)
            SCBottleneck(256, 256)
        )

        self.conv8 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            Residual_Block(128, 128),
            IBNorm(in_channels=128),
            nn.SELU(inplace=True),
            # External_attention(128)
            SCBottleneck(128, 128)
        )

        self.conv9 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            Residual_Block(64, 64),
            IBNorm(in_channels=64),
            nn.SELU(inplace=True),
            # External_attention(64)
            SCBottleneck(64, 64)
        )

        self.conv10 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            Residual_Block(32, 32),
            IBNorm(in_channels=32),
            nn.SELU(inplace=True),
            # External_attention(32)
            SCBottleneck(32, 32)
        )

        self.depth_pred = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            IBNorm(in_channels=32),
            nn.SELU(inplace=True),
            SCBottleneck(32, 32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.SELU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
        # self.dsc = DSC_Module(64, 64)
        # self.reduce2 = LayerConv(128, 64, 1, 1, 0, False)


        #####B to A
        ############################################ Rain removal network

        self.head = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, num_features, kernel_size=1, stride=1, padding=0), nn.ReLU()
        )
        # input = 64 * 160 * 240
        self.body = nn.Sequential(
            DenseRR(),
            nn.Conv2d(768, num_features, kernel_size=1, stride=1), nn.LeakyReLU(0.2)
        )
        # output = 64 * 160 * 240

        self.dgnlb = DGNL(num_features)

        self.tail = nn.Sequential(
            # nn.Conv2d(num_features, num_features, kernel_size=3, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(num_features, 32, kernel_size=4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = (x - self.mean) / self.std

        ################################## depth prediction
        d_f1 = self.conv1(x)
        d_f2 = self.conv2(d_f1)
        d_f3 = self.conv3(d_f2)
        d_f4 = self.conv4(d_f3)
        d_f5 = self.conv5(d_f4)
        d_f6 = self.conv6(d_f5)
        d_f7 = self.conv7(d_f6)
        d_f8 = self.conv8(d_f7)
        d_f9 = self.conv9(d_f8 + d_f3)
        d_f10 = self.conv10(d_f9 + d_f2)
        depth_pred = self.depth_pred(d_f10 + d_f1)

        ################################## rain removal

        # f = self.head(x)
        # output: [1, 64, 160, 240]
        f = self.body(x)
        # dsc = self.dsc(f)   # 本身就有ReLU
        # f = self.reduce2(torch.cat((f, dsc), 1))
        # output: [1, 64, 160, 240]
        # f = self.U_net(f)
        f = self.dgnlb(f, depth_pred.detach())
        # output: [1, 64, 160, 240]
        r = self.tail(f)
        # outputr: [1, 3, 320, 480]
        x = x + r
        # outputx: [1, 3, 320, 480]
        x = (x * self.std + self.mean).clamp(min=0, max=1)
        # outputx: [1, 3, 320, 480]

        if self.training:
            # return x, depth_pred,entropy_loss
            return x, depth_pred

        return x


class DiscriminativeNet(nn.Module):
    def __init__(self, width, height):
        super(DiscriminativeNet, self).__init__()
        full_channel = int(width*height/2)
        self.head = nn.Sequential(
            Constage(3, 32, 4, 2, 1),
            Constage(32, 64, 4, 2, 1),
            Constage(64, 32, 4, 2, 1),

        )
        self.body = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=5, padding=2)

        )
        self.tail = nn.Sequential(
            nn.Linear(full_channel, 128, False),
            # Constage(5, 4, 256, 128),
            #  nn.ConvTranspose2d(128,64,4,2,1),

             # nn.ConvTranspose2d(64,3,4,2),
             # nn.ConvTranspose2d(32,3,4,2),
             nn.Sigmoid()
        )

    def forward(self, result,gt):
        d1 = self.head(result)
        l1 = self.body(d1)
        input_tensor = d1 * l1
        input_tensor = input_tensor.view(input_tensor.size(0), input_tensor.size(1)*input_tensor.size(2)*input_tensor.size(3))
        fc_out = self.tail(input_tensor)
        d2 = self.head(gt)
        l2 = self.body(d2)
        input_tensor2 = d2 * l2
        input_tensor2 = input_tensor2.view(input_tensor2.size(0), input_tensor2.size(1) * input_tensor2.size(2) * input_tensor2.size(3))
        fc_out2 = self.tail(input_tensor2)
        return fc_out, fc_out2


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


class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class Residual_Block(nn.Module):
    def __init__(self, i_channel, o_channel, stride=1):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=o_channel, kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = IBNorm(o_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=o_channel, out_channels=o_channel, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn2 = IBNorm(o_channel)


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class SCConv(nn.Module):
    def __init__(self, planes, pooling_r):
        super(SCConv, self).__init__()
        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k3 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
        )
        self.k4 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(
            torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:])))  # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out)  # k3 * sigmoid(identity + k2)
        out = self.k4(out)  # k4

        return out


class SCBottleneck(nn.Module):
    # expansion = 4
    pooling_r = 4  # down-sampling rate of the avg pooling layer in the K3 path of SC-Conv.

    def __init__(self, in_planes, planes):
        super(SCBottleneck, self).__init__()
        planes = int(planes / 2)

        self.conv1_a = nn.Conv2d(in_planes, planes, 1, 1)
        self.k1 = nn.Sequential(
            nn.Conv2d(planes, planes, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

        self.conv1_b = nn.Conv2d(in_planes, planes, 1, 1)

        self.scconv = SCConv(planes, self.pooling_r)

        self.conv3 = nn.Conv2d(planes * 2, planes * 2, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x

        out_a = self.conv1_a(x)
        out_a = self.relu(out_a)

        out_a = self.k1(out_a)

        out_b = self.conv1_b(x)
        out_b = self.relu(out_b)

        out_b = self.scconv(out_b)

        out = self.conv3(torch.cat([out_a, out_b], dim=1))

        out += residual
        out = self.relu(out)

        return out



class Constage(nn.Module):
    def __init__(self, input_dims, out_dims, k_size, stride, padding=1):
        super(Constage, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_dims, out_dims, k_size, stride, padding),
            nn.SELU(),
        )

    def forward(self, x):
        output = self.layers(x)

        return output


# extracts feature from vgg16 network's 2nd and 5th pooling layer
class VGGNet(nn.Module):
    def __init__(self):
        """Select conv1_1 ~ conv5_1 activation maps."""
        super(VGGNet, self).__init__()
        self.select = ['9', '30']                           # 9和36分别代表第二和第5层的maxpooling层(19是9和36,16是9和30)
        self.vgg = models.vgg16(pretrained=True).features

    def forward(self, x):
        """Extract multiple convolutional feature maps."""
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features[0], features[1]


if __name__ == '__main__':
    # ds_net = DiscriminativeNet(512, 1024).cuda().train()
    # gs_net = GenerativeNetwork().cuda().train()
    # num = 0
    # DS_optimizer = optim.Adam(ds_net.parameters(), lr=0.0002, betas=(0.5, 0.999))
    # while num < 1000:
    #     input = torch.randn(1, 3, 512, 1024)
    #     lable = torch.randn(1, 3, 512, 1024)
    #     input = Variable(input).cuda()
    #     lable = Variable(lable).cuda()
    #     loss = ds_net(input, lable)
    #     DS_optimizer.zero_grad()
    #     loss.backward()
    #     DS_optimizer.step()
    #     print(loss)
    #     num = num + 1
    pass
