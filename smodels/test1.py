import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary



from functools import partial



from vin import PyramidVisionTransformer

pvt = PyramidVisionTransformer(patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)

summary(pvt, input_size=(1,3, 224,224))
# print(pvt)



