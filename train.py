import datetime
import os
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from TVLoss.TVLossL1 import TVLossL1
from ECLoss.ECLoss import DCLoss
import triple_transforms
from nets import GenerativeNetwork
from SN_GAN import DiscriminativeNet
from config import train_raincityscapes_path, test_raincityscapes_path
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir
# from cal_ssim import SSIM
from nets import VGGNet
from MS_SSIM_L1_loss import MS_SSIM_L1_LOSS
from misc import ReplayBuffer

# torch.cuda.set_device(0)

cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'NN2025'

args = {
    'iter_num': 300000,
    'train_batch_size': 2,
    'last_iter': 0,
    'lr': 5e-4,
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'val_freq': 5000,
    'img_size_h': 480,
    'img_size_w': 720,
    'crop_size': 512,
    'snapshot_epochs': 1000
}

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    # triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder(train_raincityscapes_path, transform=transform, target_transform=transform,
                        triple_transform=triple_transform, is_train=True)
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=0, shuffle=True)
test1_set = ImageFolder(test_raincityscapes_path, transform=transform, target_transform=transform, is_train=False)
test1_loader = DataLoader(test1_set, batch_size=2)

result_buffer = ReplayBuffer()

criterion = nn.L1Loss()
criterion_GAN = nn.MSELoss()
criterion_depth = nn.L1Loss()
# ssim = SSIM().cuda()
MS_L1 = MS_SSIM_L1_LOSS().cuda()
log_path = os.path.join(ckpt_path, exp_name, "total" + '.txt')

# Tensor = torch.cuda.FloatTensor
# target_real = Variable(Tensor(args['train_batch_size']).fill_(1.0), requires_grad=False)  # 全填充为1
# target_fake = Variable(Tensor(args['train_batch_size']).fill_(0.0), requires_grad=False)  # 全填充为0

def perceptual_loss(x, y):
    vgg = VGGNet().cuda().eval()
    c = torch.nn.MSELoss()
    # rx = netG_B2A(netG_A2B(x))  # reconstructA
    # ry = netG_A2B(netG_B2A(y))  # reconstructB

    fx1, fx2 = vgg(x)  # 提取特征
    fy1, fy2 = vgg(y)

    m1 = c(fx1, fy1)
    m2 = c(fx2, fy2)

    # loss = (m1+m2+m3+m4) * 0.00001 * 0.5
    loss = (m1 + m2) * 0.00001 * 0.5
    return loss


def patchGan(output):
    """ Performs patchGan on a 1 x 64 x 64 output of the discriminator.
    """
    output = torch.mean(output, dim=3).mean(dim=2).squeeze(1)

    return output


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data)
    elif classname.find('IBNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main():
    gs_net = GenerativeNetwork().cuda().train()
    ds_net = DiscriminativeNet().cuda().train()

    ds_net.apply(weights_init_normal)
    DS_optimizer = optim.Adam(ds_net.parameters(), lr=0.0000005, betas=(0.5, 0.999))

    GS_optimizer = optim.Adam([
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in gs_net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        gs_net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        GS_optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        GS_optimizer.param_groups[0]['lr'] = 2 * args['lr']
        GS_optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(gs_net, ds_net, GS_optimizer, DS_optimizer)


def train(gs_net, ds_net, GS_optimizer, DS_optimizer):
    curr_iter = args['last_iter']

    while True:
        train_loss_record = AvgMeter()
        train_L1loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()
        train_DSnet_loss_record = AvgMeter()
        train_GAN_loss_record = AvgMeter()
        train_DCloss_record = AvgMeter()
        train_TVloss_record = AvgMeter()

        for i, data in enumerate(train_loader):
            GS_optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            GS_optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']

            inputs, gts, dps = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            ########################################
            # Generator
            GS_optimizer.zero_grad()
            result, depth_pred = gs_net(inputs)

            # Unsupervised Loss
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            result = result.to(device)
            TV_LOSS = TVLossL1(result)
            DC_loss = DCLoss(result, 35)

            # GAN loss
            DS_fake = ds_net(result)
            DS_fake = patchGan(DS_fake)
            
            DS_real = ds_net(gts)
            DS_real = patchGan(DS_real)
            target_fake = torch.zeros(DS_fake.size()).float().cuda()  # 全填充为0
            target_real = torch.ones(DS_real.size()).float().cuda()  # 全填充为1

            loss_GAN = 0.5 * criterion_GAN(DS_fake, target_real)

            # loss_net = criterion(result, gts)
            loss_net = MS_L1(result, gts)
            loss_depth = criterion_depth(depth_pred, dps)
            # loss_ssim = 1 - ssim(result, gts)
            
            # Perceptual loss
            # loss_perceptual = perceptual_loss(result, gts)

            # loss = loss_net + loss_depth + 0.01 * (TV_LOSS + DC_loss) + loss_ssim + loss_perceptual
            loss = loss_net + 0.1 * loss_GAN + 0.01 * (TV_LOSS + DC_loss) + loss_depth
            loss.backward()
            GS_optimizer.step()

            #######################################
            # Discriminator
            DS_optimizer.zero_grad()
            result_D = result_buffer.push_and_pop(result)
            DS_fake = ds_net(result_D)
            DS_fake = patchGan(DS_fake)

            DS_real = ds_net(gts)
            DS_real = patchGan(DS_real)

            DS_loss = 0.5 * criterion_GAN(DS_fake, target_fake) + 0.5 * criterion(DS_real, target_real)

            DS_loss.backward()
            DS_optimizer.step()

            ####
            train_loss_record.update(loss.data, batch_size)
            train_L1loss_record.update(loss_net.data, batch_size)
            train_depth_loss_record.update(loss_depth.data, batch_size)
            train_DSnet_loss_record.update(DS_loss.data, batch_size)
            train_GAN_loss_record.update(loss_GAN.data, batch_size)
            train_DCloss_record.update(DC_loss, batch_size)
            train_TVloss_record.update(TV_LOSS, batch_size)

            curr_iter += 1

            log = '[iter %d], [Total_loss %.13f], [lr %.13f], [MS_L1_loss %.13f], [Depth_loss %.13f], [DS_loss %.13f],[GAN_loss %.13f], [DC_loss %.13f],[TV_loss %.13f]' % \
                  (curr_iter, train_loss_record.avg, GS_optimizer.param_groups[1]['lr'],
                   train_L1loss_record.avg, train_depth_loss_record.avg, train_DSnet_loss_record.avg,
                   train_GAN_loss_record.avg, train_DCloss_record.avg, train_TVloss_record.avg)
            print(log)
            open(log_path, 'a').write(log + '\n')


            if (curr_iter + 1) % args['snapshot_epochs'] == 0:
                torch.save(gs_net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1))))
                torch.save(GS_optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1))))

            if curr_iter > args['iter_num']:
                return

            # if loss_net <= 0.005:
            #     torch.save(gs_net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter - 1))))
            #     torch.save(GS_optimizer.state_dict(),
            #                os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter - 1))))
            #     return


if __name__ == '__main__':
    main()
