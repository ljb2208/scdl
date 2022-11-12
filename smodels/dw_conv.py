import torch
import torch.nn as nn

class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1,1,bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, W, H)
        x. self.dwconv(x)
        x = x.flatten(2).transpose(1,2)

        return x