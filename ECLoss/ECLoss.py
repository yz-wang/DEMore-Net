import torch
import numpy as np
from PIL import Image

import torch.nn as nn
from torch.nn import L1Loss, MSELoss
from torch.autograd import Variable
from torchvision import transforms
import pdb





def DCLoss(img, patch_size):
    """
    calculating dark channel of image, the image shape is of N*C*W*H
    """
    maxpool = nn.MaxPool3d((3, patch_size, patch_size), stride=1, padding=(0, patch_size//2, patch_size//2))
    dc = maxpool(0-img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()) 
     
    loss = L1Loss(size_average=True)(-dc, target)
    return 0.1*loss

def BCLoss(img, patch_size):
    """
    calculating bright channel of image, the image shape is of N*C*W*H
    """
    patch_size = 35
    dc = maxpool(img[:, None, :, :, :])
    
    target = Variable(torch.FloatTensor(dc.shape).zero_().cuda()+1) 
    loss = L1Loss(size_average=False)(dc, target)
    return loss
    
if __name__=="__main__":
    input = torch.randn(1, 3, 512, 1024)
    device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = input.to(device1)
    loss = DCLoss(input, 35)
    print(loss)
    
    # loss.backward()



    



