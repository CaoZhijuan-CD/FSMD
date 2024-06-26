import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torchvision import models

from .help_funcs import *

class UNet3plusDs(nn.Module):

    def __init__(self, in_channels=3, n_classes=1, base_model = models.resnet18(pretrained = True)):
        super(UNet3plusDs, self).__init__()
        self.in_channels = in_channels

        filters = [32, 64, 128, 256, 512, 1024]

        #-----------Encoder
        resnet1 = base_model

        # Encoder first (and second) image
        self.firstconv1 = resnet1.conv1
        self.firstbn1 = resnet1.bn1
        self.firstrelu1 = resnet1.relu
        self.firstmaxpool1 = resnet1.maxpool

        self.encoder11 = resnet1.layer1
        self.encoder12 = resnet1.layer2
        self.encoder13 = resnet1.layer3
        self.encoder14 = resnet1.layer4

        ## ----------Decoder
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks
        
        self.FAM_feat  = [320, 320, 320, 160]

        '''stage 4d'''
        # h1->64*64, hd4->16*16, Pooling 4 times
        self.h1_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        # h2->64*64, hd4->16*16, Pooling 4 times
        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        # h3->32*32, hd4->16*16, Pooling 2 times
        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        # h4->16*16, hd4->16*16, Concatenation
        self.h4_Cat_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, 3, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        # hd5->8*8, hd4->16*16, Upsample 2 times
        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[5], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4)
        self.fam4 = CAM_layers(self.FAM_feat, in_channels=160, dilation=True)

        '''stage 3d'''
        # h1->64*64, hd3->32*32, Pooling 2 times
        self.h1_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        # h2->64*64, hd3->32*32, Pooling 2 times
        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        # h3->32*32, hd3->32*32, Concatenation
        self.h3_Cat_hd3_conv = nn.Conv2d(filters[3], self.CatChannels, 3, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        # hd4->16*16, hd4->32*32, Upsample 2 times
        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        # hd5->8*8, hd4->32*32, Upsample 4 times
        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[5], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3)
        self.fam3 = CAM_layers(self.FAM_feat, in_channels=160, dilation=True)

        '''stage 2d '''
        # h1->64*64, hd2->64*64, Concatenation
        self.h1_PT_hd2_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        # h2->64*64, hd2->64*64, Concatenation
        self.h2_Cat_hd2_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        # hd3->32*32, hd2->64*64, Upsample 2 times
        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd4->16*14, hd2->64*64, Upsample 4 times
        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        # hd5->8*8, hd2->64*64, Upsample 8 times
        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[5], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        # fusion(h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2)
        self.fam2 = CAM_layers(self.FAM_feat, in_channels=160, dilation=True)

        '''stage 1d'''
        # h1->64*64, hd1->64*64, Concatenation
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[2], self.CatChannels, 3, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        # hd2->64*64, hd1->64*64, Upsample 2 times
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd3->32*32, hd1->64*64, Upsample 2 times
        self.hd3_UT_hd1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd4->16*16, hd1->64*64, Upsample 4 times
        self.hd4_UT_hd1 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, 3, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        # hd5->8*8, hd1->64*64, Upsample 16 times
        self.hd5_UT_hd1 = nn.Upsample(scale_factor=8, mode='bilinear')
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[5], self.CatChannels, 3, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        # fusion(h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1)
        self.fam1 = CAM_layers(self.FAM_feat, in_channels=160, dilation=True)

        # -------------Bilinear Upsampling--------------
        # self.upscore5 = nn.Upsample(scale_factor=32,mode='bilinear')
        self.upscore4 = nn.Upsample(scale_factor=16,mode='bilinear')
        self.upscore3 = nn.Upsample(scale_factor=8,mode='bilinear')
        self.upscore2 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.upscore1 = nn.Upsample(scale_factor=4, mode='bilinear')


        # DeepSup
        self.outconv1 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv2 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv3 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        self.outconv4 = nn.Conv2d(self.UpChannels, n_classes, 3, padding=1)
        # self.outconv5 = nn.Conv2d(filters[5], n_classes, 3, padding=1)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')

    def forward(self, t1, t2):
        # ----------Encoder_t1
        # t1: 3*256*256
        # Stage 1
        x11 = self.firstconv1(t1)
        x11 = self.firstbn1(x11)
        x11 = self.firstrelu1(x11)
        h11 = self.firstmaxpool1(x11) # 64x64x64
        # Stage 2
        h12 = self.encoder11(h11) # 64x64x64
        # Stage 3
        h13 = self.encoder12(h12) # 128x32x32
        # Stage 4
        h14 = self.encoder13(h13) # 256x16x16
        # Stage 5
        hd15 = self.encoder14(h14) # 512x8x8

        ## ----------Encoder_t2
        # t2: 3*256*256
        # Stage 1
        x21 = self.firstconv1(t2)
        x21 = self.firstbn1(x21)
        x21 = self.firstrelu1(x21)
        h21 = self.firstmaxpool1(x21) # 64x64x64 
        #Stage 2
        h22 = self.encoder11(h21) # 64x64x64
        # Stage 3
        h23 = self.encoder12(h22) # 128x32x32
        # Stage 4
        h24 = self.encoder13(h23) # 256x16x16
        # Stage 5
        hd25 = self.encoder14(h24) # 512x8x8

        h1 = torch.cat([h11, h21], 1)  # 128*64*64
        h2 = torch.cat([h12, h22], 1)  # 128*64*64
        h3 = torch.cat([h13, h23], 1)  # 256*32*32
        h4 = torch.cat([h14, h24], 1)  # 512*16*16
        hd5 = torch.cat([hd15, hd25], 1)  # 1024*8*8

        # -------------Decoder-------------
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))  # 64*16*16
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1) # 160*16*16
        hd4 = hd4 + self.fam4(hd4)

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))  # 64*32*32
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1) # 160*32*32
        hd3 = hd3 + self.fam3(hd3)

        h1_Cat_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(h1)))  # 64*64*64
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = torch.cat((h1_Cat_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1) # 160*64*64
        hd2 = hd2 + self.fam2(hd2)

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))  # 64*64*64
        hd2_Cat_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(hd2)))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = torch.cat((h1_Cat_hd1, hd2_Cat_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1) # 160*64*64
        hd1 = hd1 + self.fam1(hd1)
        
        # d5 = self.outconv5(hd5)  # 2*8*8
        # d5 = self.upscore5(d5)  # 2*256*256

        d4 = self.outconv4(hd4)  # 2*16*16
        d4 = self.upscore4(d4)  # 2*256*256

        d3 = self.outconv3(hd3)  # 2*32*32
        d3 = self.upscore3(d3)  # 2*256*256

        d2 = self.outconv2(hd2)  # 2*64*64
        d2 = self.upscore2(d2)  # 2*256*256
        
        d1 = self.outconv1(hd1)  # 2*64*64
        d1 = self.upscore1(d1)  # 2*256*256
        return d1, d2, d3, d4