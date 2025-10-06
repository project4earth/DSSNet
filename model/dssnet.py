import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class Backbone_1(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)

        self.backbone.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        init_weights(self.backbone.features[0][0])

        self.encoder_layers = list(self.backbone.features.children())
        self.encoder1 = nn.Sequential(*self.encoder_layers[0:2])
        self.encoder2 = self.encoder_layers[2]
        self.encoder3 = self.encoder_layers[3]
        self.encoder4 = nn.Sequential(*self.encoder_layers[4:6])

        del self.backbone

    def forward(self, x):
        x1 = self.encoder1(x)   #  16
        x2 = self.encoder2(x1)  #  24
        x3 = self.encoder3(x2)  #  24
        x4 = self.encoder4(x3)  # 120    
        return x2, x3, x4 # 16, 24, 24, 120

class Backbone_2(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.model = models.maxvit_t(weights="DEFAULT")

        if in_channels != 3:
            stem_layers = list(self.model.stem.children())
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
            init_weights(new_conv1)

            self.stem = nn.Sequential(
                new_conv1,
                *stem_layers[1:]
            )
        else:
            self.stem = self.model.stem

        self.block0 = self.model.blocks[0]
        self.block1 = self.model.blocks[1]

        del self.model

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.block0(x1)
        x3 = self.block1(x2)

        return x1, x2, x3
   
class AxialChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=4):
        super(AxialChannelAttention, self).__init__()
        reduced_channels = max(in_channels // ratio, 4)

        self.fc1 = nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(inplace=True)
        self.fc2 = nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

        self.apply(init_weights)

    def forward(self, x):
        _, _, H, W = x.size()

        # Average pooling axial
        avg_pool_h = F.adaptive_avg_pool2d(x, (1, W)).expand(-1, -1, H, -1)
        avg_pool_w = F.adaptive_avg_pool2d(x, (H, 1)).expand(-1, -1, -1, W)
        axial_avg_pooled = avg_pool_h + avg_pool_w

        # Max pooling axial
        max_pool_h = F.adaptive_max_pool2d(x, (1, W)).expand(-1, -1, H, -1)
        max_pool_w = F.adaptive_max_pool2d(x, (H, 1)).expand(-1, -1, -1, W)
        axial_max_pooled = max_pool_h + max_pool_w

        avg_out = self.fc2(self.relu(self.fc1(axial_avg_pooled)))
        max_out = self.fc2(self.relu(self.fc1(axial_max_pooled)))

        scale_channel = self.sigmoid(avg_out + max_out)
        return x * (1 + scale_channel) 

class AxialSpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(2),
            nn.Conv2d(2, 1, kernel_size=1, bias=False),  # Pointwise to fuse
            nn.Sigmoid()
        )

        self.apply(init_weights)

    def forward(self, x):
        _, _, H, W = x.size()

        # === Mean pooling ===
        mean_1xW = x.mean(dim=(1, 2), keepdim=True).expand(-1, -1, H, -1)  # [B, 1, H, W]
        mean_Hx1 = x.mean(dim=(1, 3), keepdim=True).expand(-1, -1, -1, W)  # [B, 1, H, W]

        # === Max pooling ===
        max_1xW, _ = x.max(dim=1, keepdim=True)                # [B, 1, H, W]
        max_1xW, _ = max_1xW.max(dim=2, keepdim=True)          # [B, 1, 1, W]
        max_1xW = max_1xW.expand(-1, -1, H, -1)                # [B, 1, H, W]

        max_Hx1, _ = x.max(dim=1, keepdim=True)                # [B, 1, H, W]
        max_Hx1, _ = max_Hx1.max(dim=3, keepdim=True)          # [B, 1, H, 1]
        max_Hx1 = max_Hx1.expand(-1, -1, -1, W)                # [B, 1, H, W]

        # === Combine all axial attention maps ===
        avg_channel = mean_1xW + mean_Hx1       # [B, 1, H, W]
        max_channel = max_1xW + max_Hx1         # [B, 1, H, W]

        fused = torch.cat([avg_channel, max_channel], dim=1) # [B, 2, H, W]
        scale_spatial = self.spatial(fused)     # [B, 1, H, W]

        return x * (1 + scale_spatial)
             
class FFM(nn.Module):
    '''Feature Fusion Module (FFM)'''
    def __init__(self, in_channels_1: int, in_channels_2: int, dropout: float=0.1):
        super(FFM, self).__init__()

        total_channels = in_channels_1 + in_channels_2
        self.refine_feat = nn.Sequential(
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.apply(init_weights)

    def forward(self, x_1, x_2):
        output = self.refine_feat(torch.cat([x_1, x_2], dim=1))
        return output
       
class AdaptiveFusionModule(nn.Module):
    def __init__(self, in_channels_1, in_channels_2, out_channels):
        super().__init__()

        self.proj_1 = nn.Sequential(
            nn.Conv2d(in_channels_1, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.proj_2 = nn.Sequential(
            nn.Conv2d(in_channels_2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.apply(init_weights)

    def forward(self, feat1, feat2):
        if feat1.size()[2:] != feat2.size()[2:]:
            feat2 = F.interpolate(feat2, size=feat1.size()[2:], mode='bilinear', align_corners=False)

        f1 = self.proj_1(feat1)
        f2 = self.proj_2(feat2)

        fused = f1 + f2
        return fused
    
class SegmentHead(nn.Module):
    def __init__(self, in_channels=128, mid_channels=128, num_classes=2, out_size=(512, 512)):
        super().__init__()
        self.out_size = out_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

        self.apply(init_weights)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dw_conv(x)
        x = self.classifier(x)
        x = F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=False)
        return x

class DSSNet(nn.Module):
    def __init__(self, in_channels_1: int = 2, in_channels_2: int = 4, num_classes: int = 2, dropout: float=0.1, out_size=(512, 512)):
        super().__init__()
        self.backbone_1 = Backbone_1(in_channels_1)
        self.backbone_2 = Backbone_2(in_channels_2)
        self.num_classes = num_classes

        b_chan_1 = [24, 40, 112]
        b_chan_2 = [64, 64, 128]
        mid_chan = [32, 120]

        self.afm_low = AdaptiveFusionModule(b_chan_1[0], b_chan_2[0], mid_chan[0])
        self.afm_high = AdaptiveFusionModule(b_chan_1[2], b_chan_2[2], mid_chan[1])

        self.asa = AxialSpatialAttention()
        self.aca = AxialChannelAttention(mid_chan[1])

        self.feat_fusion = FFM(mid_chan[0], mid_chan[1], dropout=dropout)
        self.classifier = SegmentHead(in_channels=mid_chan[0] + mid_chan[1], mid_channels=mid_chan[1], num_classes=self.num_classes, out_size=out_size)

    def forward(self, x_1, x_2):
        x_2 = F.interpolate(x_2, size=(224, 224), mode="bilinear", align_corners=False)

        low_level_1, _, high_level_1 = self.backbone_1(x_1)
        low_level_2, _, high_level_2 = self.backbone_2(x_2)

        fused_low = self.afm_low(low_level_1, low_level_2)
        fused_high = self.afm_high(high_level_1, high_level_2)

        fused_low = self.asa(fused_low)
        fused_high = self.aca(fused_high)
        fused_high = F.interpolate(fused_high, scale_factor=4, mode='bilinear', align_corners=False) 

        x = self.feat_fusion(fused_high, fused_low)
        return self.classifier(x)