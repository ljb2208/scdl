from turtle import forward
import torch
import torch.nn as nn
from einops import rearrange

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
         x = rearrange(x, "b c h w -> b h w c")
         x = super().forward(x)
         x = rearrange(x, "b h w c -> b c h w")
         return x