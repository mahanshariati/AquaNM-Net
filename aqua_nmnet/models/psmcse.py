import torch
import torch.nn as nn

from aqua_nmnet.models.color_spaces_utils import rgb2hsi, rgb2lab


class WB(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)
        self.tanh=nn.Tanh()

    def forward(self, x):
        out1 = self.avgpool(x)
        out2 = self.conv(x)
        out3 = self.maxpool(x)
        out4 = self.conv1(x)
        out =self.conv2(out2 / (out1 + 1e-5)+self.tanh(self.conv4(x))) + self.conv3(out4 / (out3 + 1e-5)+self.tanh(self.conv5(x)))
        return out


class RGBPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,kernel_size=1,stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self,x):
        min = -self.maxpool(-x)
        max = self.maxpool(x)
        out = (x - min) / (max - min + 1e-5)
        out = self.conv(out)
        return out
        

class LabPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3,3,kernel_size=1,stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self,x):
        rgb = torch.sigmoid(x)
        x = rgb2lab(rgb)
        min = -self.maxpool(-x)
        max = self.maxpool(x)
        out = (x - min) / (max - min + 1e-5)
        out = self.conv(out)
        return out


class HSIPR(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=1, stride=1)
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        rgb = torch.sigmoid(x)
        out = rgb2hsi(rgb)
        h, si = torch.split(out, [1, 2], dim=1)
        minsi = -self.maxpool(-si)
        maxsi = self.maxpool(si)
        si = (si - minsi) / (maxsi - minsi + 1e-5)
        si = si + self.conv(si)
        out = torch.cat((h, si), dim=1)
        return out


class PSMCSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=1,padding=1)
        self.tanh=nn.Tanh()
        self.step1=WB()
        self.step2=RGBPR()
        self.step3=LabPR()
        self.step4=HSIPR()
        self.relu=nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.step1(x)

        out1 = self.step2(out) + self.tanh(self.conv1(out))
        
        l_rgb = 0.5 * (self.tanh(self.conv2(out)) + 1.0)
        lab_residual = rgb2lab(l_rgb, clip_rgb=True)
        lab_residual = torch.cat(
            [
                lab_residual[:, 0:1] / 100.0,
                (lab_residual[:, 1:3] + 128.0) / 256.0,
            ],
            dim=1
        )
        out2 = self.step3(out) + lab_residual
                
        hsi_rgb = 0.5 * (self.tanh(self.conv3(out)) + 1.0)
        out3 = self.step4(out) + rgb2hsi(hsi_rgb)

        return out1, out2, out3
