import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.nn import Module, Conv2d, Parameter

def l2_norm(x, eps=1e-6):
    norm = torch.norm(x, p=2, dim=-2, keepdim=True)
    norm = norm.clamp_min(eps)
    return x / norm

class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x

class Attention(Module):
    def __init__(self, in_places, scale=8, eps=1e-6):
        super(Attention, self).__init__()
        self.gamma = Parameter(torch.zeros(1))
        self.in_places = in_places
        self.l2_norm = l2_norm
        self.eps = eps

        self.query_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_places, out_channels=in_places // scale, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_places, out_channels=in_places, kernel_size=1)

    def forward(self, x):
        if not torch.isfinite(x).all():
            print("[Warning] Input to Attention contains NaN/Inf")
        B, C, H, W = x.shape
        Q = self.query_conv(x).view(B, -1, H * W)
        K = self.key_conv(x).view(B, -1, H * W)
        V = self.value_conv(x).view(B, -1, H * W)
   
        Q = self.l2_norm(Q).permute(0, 2, 1)
        K = self.l2_norm(K)

        einsum_result = torch.einsum("bnc, bc->bn", Q, torch.sum(K, dim=-1))
        denom = W * H + einsum_result
        denom = denom.clamp(min=1e-6)  
        tailor_sum = 1.0 / denom

        matrix = torch.einsum('bmn, bcn->bmc', K, V)
        value_sum = torch.einsum("bcn->bc", V).unsqueeze(-1).expand(-1, C, W * H)
        matrix_sum = value_sum + torch.einsum("bnm, bmc->bcn", Q, matrix)

        weight_value = torch.einsum("bcn, bn->bcn", matrix_sum, tailor_sum)
        weight_value = weight_value.view(B, C, H, W)

        out = self.gamma * weight_value
        out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)  
        return out.contiguous()

class AttentionAggregationModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(AttentionAggregationModule, self).__init__()
        self.convblk = ConvBnRelu(in_chan, out_chan, ksize=1, stride=1, pad=0)
        self.conv_atten = Attention(out_chan)

    def forward(self, s5, s4, s3, s2):
        for name, feat in zip(["s5", "s4", "s3", "s2"], [s5, s4, s3, s2]):
            if not torch.isfinite(feat).all():
                print(f"[Warning] NaN/Inf in {name} before concat")

        fcat = torch.cat([s5, s4, s3, s2], dim=1)
        feat = self.convblk(fcat)
        atten = self.conv_atten(feat)
        feat_out = atten + feat
        return feat_out

class Conv3x3GNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        self.upsample = upsample
        group = min(32, out_channels)
        if out_channels % group != 0:
            group = 1  # fallback to LayerNorm-like
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(group, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.block(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        return x

class FPNBlock(nn.Module):
    def __init__(self, pyramid_channels, skip_channels):
        super().__init__()
        self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)

    def forward(self, x):
        x, skip = x
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        skip = self.skip_conv(skip)

        x = x + skip
        return x

class SegmentationBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_upsamples=0):
        super().__init__()

        blocks = [
            Conv3x3GNReLU(in_channels, out_channels, upsample=bool(n_upsamples))
        ]

        if n_upsamples > 1:
            for _ in range(1, n_upsamples):
                blocks.append(Conv3x3GNReLU(out_channels, out_channels, upsample=True))

        self.block = nn.Sequential(*blocks)

    def forward(self, x):
        return self.block(x)

class A2FPN(nn.Module):
    def __init__(
        self,
        in_channels=11,
        num_classes=2,
        encoder_channels=[512, 256, 128, 64],
        pyramid_channels=64,
        segmentation_channels=64,
        dropout=0.2,
    ):
        super().__init__()
        self.name = 'A2FPN'
        resnet = models.resnet18(weights='DEFAULT')

        if in_channels != 3:
            self.layer0_custom = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu
            )
        else:
            self.layer0_custom = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.layer_down0 = self.layer0_custom
        self.layer_down1 = nn.Sequential(resnet.maxpool, resnet.layer1)  # H/4
        self.layer_down2 = resnet.layer2                                 # H/8
        self.layer_down3 = resnet.layer3                                 # H/16
        self.layer_down4 = resnet.layer4                                 # H/32

        self.conv1 = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1, 1))

        self.p4 = FPNBlock(pyramid_channels, encoder_channels[1])
        self.p3 = FPNBlock(pyramid_channels, encoder_channels[2])
        self.p2 = FPNBlock(pyramid_channels, encoder_channels[3])

        self.s5 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=3)
        self.s4 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=2)
        self.s3 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=1)
        self.s2 = SegmentationBlock(pyramid_channels, segmentation_channels, n_upsamples=0)

        self.attention = AttentionAggregationModule(segmentation_channels * 4, segmentation_channels * 4)
        self.final_conv = nn.Conv2d(segmentation_channels * 4, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x):
        c1 = self.layer_down0(x)
        c2 = self.layer_down1(c1)
        c3 = self.layer_down2(c2)
        c4 = self.layer_down3(c3)
        c5 = self.layer_down4(c4)

        p5 = self.conv1(c5)
        p4 = self.p4([p5, c4])
        p3 = self.p3([p4, c3])
        p2 = self.p2([p3, c2])

        s5 = self.s5(p5)
        s4 = self.s4(p4)
        s3 = self.s3(p3)
        s2 = self.s2(p2)

        out = self.dropout(self.attention(s5, s4, s3, s2))
        out = self.final_conv(out)
        out = F.interpolate(out, scale_factor=4, mode='bilinear', align_corners=True)

        return out