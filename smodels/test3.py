import torch
import torch.nn as nn
import math

from torchinfo import summary

from overlap_patch_embedding import OverlapPatchEmbedding
from self_attention import SelfAttention
from model import PatchEmbed
from model import Attention
from model import MixFFN
from model import Block
from model import DepthModel


x = torch.randn((1, 64, 32768))

c1 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=1, stride=1)


y = c1.forward(x)

print(y.shape)

y1 = y.reshape(B, C, -1)

print(y1.shape)


#depth.forward(x)

# pe = PatchEmbed(img_size=224, embed_dim=64, patch_size=4)
# y, HW =pe.forward(x)

# print("Here")
# print(y.shape)
# print(HW)


# block = Block()
# x_ = block.forward(y, HW[0], HW[1])

# # x = torch.randn((1, 3136, 64))
# attn = Attention(64, sr_ratio=2)
# x_ = attn.forward(y, HW[0], HW[1])

# mix = MixFFN()
# y_ = mix.forward(x_)

#print("Here2")
#print(x.shape)


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