import torch
import torch.nn as nn
from torchvision.models import inception_v3, resnet50, ResNet50_Weights

import torch
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, 1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out * x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out * x

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        return out

class InceptionV3_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3_CBAM, self).__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception.fc = nn.Linear(2048, num_classes)

        self.cbam1 = CBAM(64)  # Update the number of channels to match the output of Conv2d_2b_3x3
        self.cbam2 = CBAM(192)  # Update the number of channels to match the output of Conv2d_4a_3x3
        self.cbam3 = CBAM(768)
        self.cbam4 = CBAM(2048)

    def forward(self, x):
        # Inception V3 layers
        x = self.inception.Conv2d_1a_3x3(x)
        x = self.inception.Conv2d_2a_3x3(x)
        x = self.inception.Conv2d_2b_3x3(x)
        x = self.inception.maxpool1(x)
        x = self.cbam1(x)

        x = self.inception.Conv2d_3b_1x1(x)
        x = self.inception.Conv2d_4a_3x3(x)
        x = self.inception.maxpool2(x)
        x = self.cbam2(x)

        x = self.inception.Mixed_5b(x)
        x = self.inception.Mixed_5c(x)
        x = self.inception.Mixed_5d(x)
        x = self.inception.Mixed_6a(x)
        x = self.inception.Mixed_6b(x)
        x = self.inception.Mixed_6c(x)
        x = self.inception.Mixed_6d(x)
        x = self.inception.Mixed_6e(x)
        x = self.cbam3(x)

        x = self.inception.Mixed_7a(x)
        x = self.inception.Mixed_7b(x)
        x = self.inception.Mixed_7c(x)
        x = self.inception.avgpool(x)
        x = self.cbam4(x)

        x = torch.flatten(x, 1)
        x = self.inception.dropout(x)
        x = self.inception.fc(x)

        return x

import os
import torch
import torch.nn as nn
from torchvision.models import resnet50

from pretrainedmodels import xception, pretrained_settings

current_script_path = os.path.dirname(os.path.abspath(__file__))
root_path = os.path.dirname(current_script_path)
data_dir = os.path.join(root_path, "data")

pretrained_settings['xception']['imagenet']['url'] = "file://" + os.path.join(data_dir, "xception-43020ad28.pth")

class Xception_ResNet50(nn.Module):
    def __init__(self):
        super(Xception_ResNet50, self).__init__()
        self.xception = xception(pretrained='imagenet')
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Fusion layers
        self.fusion1 = nn.Conv2d(2048, 2048, kernel_size=1)
        self.fusion2 = nn.Conv2d(2048, 2048, kernel_size=1)

        # Classification layers
        self.fc1 = nn.Linear(409600, 2048)  # Update the input size to match the flattened feature tensor
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, x):
        # Xception forward pass
        x_xception = self.xception.features(x)

        # ResNet50 forward pass
        x_resnet = self.resnet.conv1(x)
        x_resnet = self.resnet.bn1(x_resnet)
        x_resnet = self.resnet.relu(x_resnet)
        x_resnet = self.resnet.maxpool(x_resnet)
        x_resnet = self.resnet.layer1(x_resnet)
        x_resnet = self.resnet.layer2(x_resnet)
        x_resnet = self.resnet.layer3(x_resnet)
        x_resnet = self.resnet.layer4(x_resnet)

        # Adaptive average pooling
        x_xception = nn.AdaptiveAvgPool2d(output_size=(10, 10))(x_xception)
        x_resnet = nn.AdaptiveAvgPool2d(output_size=(10, 10))(x_resnet)

        # Feature fusion
        x_xception = self.fusion1(x_xception)
        x_resnet = self.fusion2(x_resnet)
        x = torch.cat((x_xception, x_resnet), dim=1)

        # Classification
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)

        return x