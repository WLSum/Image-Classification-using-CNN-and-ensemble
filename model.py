import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import mobilenet_v2, vgg16_bn


class ChannelAttention(nn.Module):
    """ Channel Attention emphasizes informative channels, by compressing spatial dimensions using global pooling, 
        then uses a small FC network to learn weights """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        w = self.pool(x).view(batch, channel)
        w = self.fc(w).view(batch, channel, 1, 1)
        return x * w.expand_as(x)
    
class SpatialAttention(nn.Module):
    """ Spatial Attention highlights important regions in the feature map,
        using average-pooling and max-pooling across channels to infer spatial importance"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True) 
        max_out = torch.max(x, dim=1, keepdim=True)[0]  
        combined = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(combined)) 
        return x * attn 

class CBAM(nn.Module):
    """ CBAM = Channel Attention + Spatial Attention"""
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x

class SPPLayer(nn.Module):
    """Spatial Pyramid Pooling combines features from multiple spatial levels"""
    def __init__(self, levels=[1, 2, 4]):
        super(SPPLayer, self).__init__()
        self.levels = levels

    def forward(self, x):
        batch, channel, height, width = x.size()
        spp_features = []
        for level in self.levels:
            pooled = F.adaptive_max_pool2d(x, output_size=(level, level))
            spp_features.append(pooled.view(batch, -1))   # Flatten each pooled output
        return torch.cat(spp_features, dim=1)             # Concatenate features from all levels

  
class BuiltCNN(nn.Module):
    """Build a CNN model with batch normalization (BN), residual learning, CBAM and SPP"""
    def __init__(self, num_classes):
        super(BuiltCNN, self).__init__()

        # Conv Block 1 + BN + CBAM
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.cbam1 = CBAM(32)

        # Conv Block 2 + BN + CBAM + residual
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.cbam2 = CBAM(64)
        self.res_proj1 = nn.Conv2d(32, 64, kernel_size=1)


        # Conv Block 3 + BN + CBAM + residual
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.cbam3 = CBAM(128)
        self.res_proj2 = nn.Conv2d(64, 128, kernel_size=1)


        # Conv Block 4 + BN + CBAM + residual
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.cbam4 = CBAM(256)
        self.res_proj3 = nn.Conv2d(128, 256, kernel_size=1)
        
        
        # Conv Block 5 + BN + CBAM + residual
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.cbam5 = CBAM(512)
        self.res_proj4 = nn.Conv2d(256, 512, kernel_size=1)
        
        # Apply SPP to extract multi-scale features
        self.spp = SPPLayer(levels=[1, 2, 4]) 
        
        # Partial global average pooling
        self.gap = nn.AdaptiveAvgPool1d(1)  

#       # Final classification layer 
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 , num_classes)
        )


    def forward(self, x):
        # Conv Block 1 
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = self.cbam1(x)
       
        # Conv Block 2
        residual = x
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = self.cbam2(x)
        residual = F.max_pool2d(self.res_proj1(residual), 2)
        x = x + residual

        # Conv Block 3
        residual = x
        x = F.max_pool2d(F.relu(self.bn3(self.conv3(x))), 2)
        x = self.cbam3(x)
        residual = F.max_pool2d(self.res_proj2(residual), 2)
        x = x + residual

        # Conv Block 4
        residual = x
        x = F.max_pool2d(F.relu(self.bn4(self.conv4(x))), 2)
        x = self.cbam4(x)
        residual = F.max_pool2d(self.res_proj3(residual), 2)
        x = x + residual
        
        # Conv Block 5
        residual = x
        x = F.max_pool2d(F.relu(self.bn5(self.conv5(x))), 2)
        x = self.cbam5(x)
        residual = F.max_pool2d(self.res_proj4(residual), 2)
        x = x + residual
        
        # SPP + partial GAP 
        x = self.spp(x)
        x = x.view(x.size(0), 512, -1)
        x = self.gap(x).squeeze(-1) 

        # Final classification
        x = self.fc(x)
        return x
    

class MobileNetV2_Custom(nn.Module):
    """Custom MobileNetV2 with SPP and Partial GAP"""
    def __init__(self, num_classes):
        super(MobileNetV2_Custom, self).__init__()
        base_model = mobilenet_v2(weights='DEFAULT')
        self.features = base_model.features  
        
        self.spp = SPPLayer([1, 2, 4])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1280 , num_classes)
        )

    def forward(self, x):
        x = self.features(x)  
        x = self.spp(x)
        x = x.view(x.size(0), 1280, -1)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x


class VGG16_Custom(nn.Module):
    """Custom VGG16 with SPP and Partial GAP"""
    def __init__(self, num_classes):
        super(VGG16_Custom, self).__init__()

        base_model = vgg16_bn(weights='DEFAULT')
        self.features = base_model.features

        self.spp = SPPLayer([1, 2, 4])
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 , num_classes) 
        )

    def forward(self, x):
        x = self.features(x)
        x = self.spp(x)
        x = x.view(x.size(0), 512, -1)
        x = self.gap(x).squeeze(-1)
        x = self.fc(x)
        return x