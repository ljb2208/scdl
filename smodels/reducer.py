import torch
import torch.nn as nn
from layer_norm import LayerNorm2d

class Reducer(nn.Module):
    def __init__(self, channels, reduction_ratio):
        super().__init__()

        self.conv = nn.Conv2d(channels, channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        #self.norm = LayerNorm2d(channels)

    def forward(self, x):
        x = self.conv(x)        
        #x = self.norm(x)
        

        return x
