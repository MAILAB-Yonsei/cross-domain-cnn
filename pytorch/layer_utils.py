import torch
import torch.nn as nn

def roll(x, shift, dim=-1):
    if shift == 0:
        return x

    elif shift < 0:
        shift = -shift
        gap = x.index_select(dim, torch.arange(shift).cuda())
        return torch.cat([x.index_select(dim, torch.arange(shift, x.size(dim)).cuda()), gap], dim=dim)

    else:
        shift = x.size(dim) - shift
        gap = x.index_select(dim, torch.arange(shift, x.size(dim)).cuda())
        return torch.cat([gap, x.index_select(dim, torch.arange(shift).cuda())], dim=dim)

def ifft2(input_):
    return torch.ifft(input_.permute(0,2,3,1),2).permute(0,3,1,2)

def fft2(input_):
    return torch.fft(input_.permute(0,2,3,1),2).permute(0,3,1,2)

def ifft1(input_, axis):
    if   axis == 1:
        return torch.ifft(input_.permute(0,2,3,1),1).permute(0,3,1,2)
    elif axis == 0:
        return torch.ifft(input_.permute(0,3,2,1),1).permute(0,3,2,1)

def fft1(input_, axis):
    if   axis == 1:
        return torch.fft(input_.permute(0,2,3,1),1).permute(0,3,1,2)
    elif axis == 0:
        return torch.fft(input_.permute(0,3,2,1),1).permute(0,3,2,1)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight.data, a=0.1, nonlinearity='leaky_relu')
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def fftshift(x, dim):
    shift = x.size(dim) // 2
    return roll(x, shift, dim)

def fftshift2(x):
    return fftshift(fftshift(x, -1),-2)

def GenConvBlock(n_conv_layers, in_chan, out_chan, feature_maps):
    conv_block = [nn.Conv2d(in_chan, feature_maps, 3, 1, 1),
                  nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    for _ in range(n_conv_layers - 2):
        conv_block += [nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
                       nn.LeakyReLU(negative_slope=0.1, inplace=True)]
    return nn.Sequential(*conv_block, nn.Conv2d(feature_maps, out_chan, 3, 1, 1))

def GenUnet():
    return 

def GenFcBlock(feat_list=[512, 1024, 1024, 512]):
    FC_blocks = []
    len_f = len(feat_list)
    for i in range(len_f - 2):
        FC_blocks += [nn.Linear(feat_list[i], feat_list[i + 1]),
                      nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        
    return nn.Sequential(*FC_blocks, nn.Linear(feat_list[len_f - 2], feat_list[len_f - 1]))

def DataConsist(input_, k, m, is_k=False):
    if is_k:
        return input_ * m + k * (1 - m)
    else:
        input_p = input_.permute(0,2,3,1); k_p = k.permute(0,2,3,1); m_p = m.permute(0,2,3,1)
        return torch.ifft(torch.fft(input_p, 2) * m_p + k_p * (1 - m_p), 2).permute(0,3,1,2)