import torch
import torch.nn as nn
import math

from overlap_patch_embedding import OverlapPatchEmbedding
from self_attention import SelfAttention
from model import PatchEmbed


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = [img_size, img_size]
        patch_size = [patch_size, patch_size]

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)


    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W



x = torch.randn((1, 3, 224,224))

pe = PatchEmbed(img_size=224, embed_dim=64, patch_size=4)
y, HW =pe.forward(x)

print("Here")
print(y.shape)
print(HW)

# ope = OverlapPatchEmbedding(img_size=224, in_channels=3, embed_dim=64, patch_size=16, overlap_size=4)
# y, X, H = ope.forward(x)

# print("Here2")
# print(y.shape)

# # sa = SelfAttention(64, 64, 4)
# # sa.forward(y, X, H)

# op = OverlapPatchEmbed(img_size=224, patch_size=16, stride=16, embed_dim=64)
# y, X, H = op.forward(x)

# print("Here3")
# print(y.shape)
# print(X)
# print(H)