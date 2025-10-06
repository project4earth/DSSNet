import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AtrousConvolution(nn.Module):
    def __init__(self, input_channels, kernel_size, pad, dilation_rate, output_channels=128):
        super(AtrousConvolution, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size, padding=pad,
                              dilation=dilation_rate, bias=False)
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv_1x1 = AtrousConvolution(in_channels, 1, 0, 1, out_channels)
        self.conv_3x3_1 = AtrousConvolution(in_channels, 3, 6, 6, out_channels)
        self.conv_3x3_2 = AtrousConvolution(in_channels, 3, 12, 12, out_channels)
        self.conv_3x3_3 = AtrousConvolution(in_channels, 3, 18, 18, out_channels)

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.final_conv = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.final_batchnorm = nn.BatchNorm2d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv_1x1(x)
        x2 = self.conv_3x3_1(x)
        x3 = self.conv_3x3_2(x)
        x4 = self.conv_3x3_3(x)
        x5 = self.image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.final_conv(x)
        x = self.final_batchnorm(x)
        x = self.final_relu(x)
        return x
    
class ResNet18(nn.Module):
    def __init__(self, in_channels):
        super(ResNet18, self).__init__()
        resnet = models.resnet18(weights='DEFAULT')
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3

        del resnet

    def forward(self, x):
        x = self.stem(x)      # 256x256
        x1 = self.layer1(x)   # 128x128
        x2 = self.layer2(x1)  # 64x64
        x3 = self.layer3(x2)  # 32x32

        return x1, x3

class DeepLabv3Plus(nn.Module):
    def __init__(self, in_channels=11, num_classes=2):
        super(DeepLabv3Plus, self).__init__()

        self.backbone = ResNet18(in_channels)

        self.aspp = ASPP(in_channels=256, out_channels=128)
        self.conv1x1 = nn.Conv2d(64, 48, kernel_size=1, bias=False)
        self.conv1x1_bn = nn.BatchNorm2d(48)
        self.conv1x1_relu = nn.ReLU(inplace=True)

        self.concat_conv = nn.Sequential(
            nn.Conv2d(176, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        x_low_level, x = self.backbone(x)

        x = self.aspp(x)
        x = F.interpolate(x, size=x_low_level.size()[2:], mode='bilinear', align_corners=True)

        x_low_level = self.conv1x1(x_low_level)
        x_low_level = self.conv1x1_bn(x_low_level)
        x_low_level = self.conv1x1_relu(x_low_level)

        x = torch.cat((x, x_low_level), dim=1)
        x = self.concat_conv(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        x = self.classifier(x)
        return x