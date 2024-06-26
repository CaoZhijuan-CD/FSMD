import torch
import torch.nn.functional as F
import math
import einops
from einops import rearrange
from torch import nn
from torch.nn import Module, Conv2d, Parameter, Softmax
from torch.nn import init


def last_activation(name):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'softmax':
        return nn.Softmax(dim=1)
    elif name == 'logsoftmax':
        return nn.LogSoftmax(dim = 1)
    elif name == 'no':
        return nn.Identity()
        
#--------------SUNet
def relu():
    return nn.ReLU(inplace=True)

class BasicConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, pad='zero', bn=False, act=False, **kwargs):
        super().__init__()
        self.seq = nn.Sequential()
        if kernel>=2:
            self.seq.add_module('_pad', getattr(nn, pad.capitalize()+'Pad2d')(kernel//2))
        self.seq.add_module('_conv', nn.Conv2d(
            in_ch, out_ch, kernel,
            stride=1, padding=0,
            bias=not bn,
            **kwargs
        ))
        if bn:
            self.seq.add_module('_bn', nn.BatchNorm2d(out_ch))
        if act:
            self.seq.add_module('_act', relu())

    def forward(self, x):
        return self.seq(x)


class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad='zero', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, pad=pad, bn=bn, act=act, **kwargs)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, bn=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, bn=True, act=False)
    
    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))


class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, bn=True, act=True):
        super().__init__()
        self.deconv =  nn.ConvTranspose2d(in_ch2, in_ch2, kernel_size=2, padding=0, stride=2)
        self.conv_feat = ResBlock(in_ch1+in_ch2, in_ch2)
        self.conv_out = Conv3x3(in_ch2, out_ch, bn=bn, act=act)

    def forward(self, x1, x2):
        x2 = self.deconv(x2)
        pl = 0
        pr = x1.size(3)-x2.size(3)
        pt = 0
        pb = (x1.size(2)-x2.size(2))
        x2 = F.pad(x2, (pl, pr, pt, pb), 'replicate')
        x = torch.cat((x1, x2), dim=1)
        x = self.conv_feat(x)
        return self.conv_out(x)
    

#----------unet3+
# 网络参数初始化
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


class unetConv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(unetConv2, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        identity = x
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x + identity)
        return output
# ----------变化感知模块
class CAM_layers(nn.Module):
    def __init__(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        super(CAM_layers, self).__init__()
        if dilation:
            d_rate = 3
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
        self.conv1 = nn.Sequential(*layers)
    def forward(self, x):
        return self.conv1(x)
