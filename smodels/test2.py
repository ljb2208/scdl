import torch
import torch.nn as nn
import math

from torchinfo import summary

from overlap_patch_embedding import OverlapPatchEmbedding
from self_attention import SelfAttention
from model import PatchEmbed
from model import Attention
from model import AttentionOrig
from model import MixFFN
from model import Block
from model import DepthEncoder
from model import DecoderBlock
from model import SegAttention
from model import SegOverlapPatchEmbed
from model import Decoder



# x = torch.randn((1, 320, 16, 32))
# attn = torch.rand((1,250,32,64))

# dec = Decoder()
# y = dec.forward(x, attn)

# print(y.shape)

# ope = SegOverlapPatchEmbed(img_size=[512,1024], embed_dim=64)
# # ope = PatchEmbed(img_size=[512,1024], embed_dim=64, patch_size=7)

# y = ope.forward(x)

# print(y[0].shape)
# print(y[1])
# print(y[2])


# torch.Size([1, 1, 32768, 64])
# torch.Size([1, 1, 512, 64])
# torch.Size([1, 1, 32768, 512])
# torch.Size([1, 1, 512, 64])


# x = torch.randn([1, 1, 32768, 512])

# y = torch.max(x, 3)

# print(y[0].shape)

# x = torch.randn((1,64, 32768))

# y = torch.mean(x, 2)
# print(y.shape)
# print(y)
# y = y.expand(1,1,64)

# y = torch.tile(y, (1,64,1))
# print(y.shape)
# print(y)

# x = torch.randn((1,32768, 64))
# x = torch.randn(1,3,512,1024)

# ffn = MixFFN(num_embeddings=64)
# y = ffn.forward(x, 512, 1024)


# model = DepthModel(img_size=[512,1024])
# y = model.forward(x)


x = torch.randn((1, 3, 512,1024))
dp = DepthEncoder(img_size=[512,1024])
dec = DecoderBlock(img_size=[512,1024])
y = dp.forward(x)

z = dec.forward(y)

print(z.shape)

# blk =Block(num_embeddings=64, hidden_features=256, sr_ratio=8)
# y = blk.forward(x, 128, 256)

for i in range(len(y)):
    print(y[i].shape)
# print(y.shape)

#att = Attention(num_embeddings=64, sr_ratio=4, num_heads=1)
#y = att.forward(x, 128, 256)

# att = SegAttention(dim=64, sr_ratio=8, num_heads=1)
# z = att.forward(y[0], y[1], y[2])

# print(z.shape)

#depth.forward(x)

# pe = PatchEmbed(img_size=224, embed_dim=64, patch_size=4)
# y, HW =pe.forward(x)

# print("Here")
print(y.shape)
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