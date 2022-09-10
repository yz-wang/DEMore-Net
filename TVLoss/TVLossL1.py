import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torch.nn.functional import l1_loss
from torch.nn import Conv2d
import torch.nn as nn

# img must be variable with grad and of dim N*C*W*H
def TVLossL1(img):
    hor = grad_conv_hor()(img)
    vet = grad_conv_vet()(img)
    target = torch.autograd.Variable(torch.FloatTensor(img.shape).zero_().cuda())
    loss_hor = l1_loss(hor, target, size_average=False)
    loss_vet = l1_loss(vet, target, size_average=False)
    loss = loss_hor+loss_vet
    return 0.00001 * loss


# horizontal gradient, the input_channel is default to 3
def grad_conv_hor():
	grad = Conv2d(3, 3, (1, 3), stride=1, padding=(0, 1))
	
	weight = np.zeros((3, 3, 1, 3))
	for i in range(3):
		weight[i, i, :, :] = np.array([[-1, 1, 0]])
	weight = torch.FloatTensor(weight).cuda()
	weight = nn.Parameter(weight, requires_grad=False)
	bias = np.array([0, 0, 0])
	bias = torch.FloatTensor(bias).cuda()
	bias = nn.Parameter(bias, requires_grad=False)
	grad.weight = weight
	grad.bias = bias
	return  grad

# vertical gradient, the input_channel is default to 3
def grad_conv_vet():
	grad = Conv2d(3, 3, (3, 1), stride=1, padding=(1, 0))
	weight = np.zeros((3, 3, 3, 1))
	for i in range(3):
		weight[i, i, :, :] = np.array([[-1, 1, 0]]).T
	weight = torch.FloatTensor(weight).cuda()
	weight = nn.Parameter(weight, requires_grad=False)
	bias = np.array([0, 0, 0])
	bias = torch.FloatTensor(bias).cuda()
	bias = nn.Parameter(bias, requires_grad=False)
	grad.weight = weight
	grad.bias = bias
	return  grad
    

if __name__ == "__main__":
    # img = Image.open('1.jpg')
    # img = transforms.ToTensor()(img)[None, :, :, :]
    # img = torch.autograd.Variable(img, requires_grad=True)
    input = torch.randn(1, 3, 512, 1024)
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input=input.to(device1)
    loss = TVLossL1(input)
    print(loss)
