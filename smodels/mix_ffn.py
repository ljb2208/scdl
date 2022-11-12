import torch
import torch.nn as nn

from dw_conv import DWConv

class MixFFN(nn.Module):
    def __init(self, in_features, out_features, hidden_features):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=in_features)
        self.dwconv = DWConv(hidden_features)
        
        self.bn2 = nn.BatchNorm2d(num_features=in_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=1)

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.bn1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x




