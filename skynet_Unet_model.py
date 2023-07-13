from torch import nn
import torch
import torch.nn.functional as F

class double_conv(nn.Module):

    def __init__(self, in_ch, out_ch,bn=False):
        super(double_conv, self).__init__()
        self.conv= nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        x=self.conv(x)
        return x

class inputconv(nn.Module):
    def __init__(self, in_ch, out_ch,bn=False):
        super(inputconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch,bn)

    def forward(self, x):
        x = self.conv(x)
        return x

class outputconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outputconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down_layers(nn.Module):
    def __init__(self, in_ch, out_ch,bn=False):
        super(down_layers, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch,bn),
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up_layers(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=False,bn=False):
        super(up_layers, self).__init__()
        self.bilinear=bilinear

        if self.bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch,in_ch//2,1),)

        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch,bn)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,diffY // 2, diffY - diffY // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class SkyNet_UNet(nn.Module):
    def __init__(self,n_channels,layer_nums,features_root=64, output_channel=3, bn=False):
        super(SkyNet_UNet,self).__init__()
        self.inc = inputconv(n_channels, 64,bn)
        self.down1 = down_layers(64, 128,bn)
        self.down2 = down_layers(128, 256,bn)
        self.down3 = down_layers(256, 512,bn)
        self.up1 = up_layers(512, 256)
        self.up2 = up_layers(256, 128)
        self.up3 = up_layers(128, 64)
        self.outc = outputconv(64, output_channel)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)

        return x