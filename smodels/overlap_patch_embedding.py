import torch
import torch.nn as nn

class OverlapPatchEmbedding(nn.Module):
    def __init__(self, img_size, in_channels, embed_dim, patch_size, overlap_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=overlap_size, padding=patch_size // 2, bias=False)
        # self.bn1 = nn.BatchNorm1d(num_features=(in_channels * embed_dim / 2))
        self.bn1 = nn.BatchNorm1d(num_features=int(img_size * embed_dim / 2))
        # self.bn1 = nn.LayerNorm(out_channels)
        #self.bn1 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv1(x)        
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.bn1(x)

        return x, H, W