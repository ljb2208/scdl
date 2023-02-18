
from matplotlib.cbook import simple_linear_interpolation
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        # assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
        #     f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                                padding=(patch_size[0] // 2, patch_size[1] //2))        
        self.norm = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2)
        x = self.norm(x)
        # x = self.norm(x).transpose(1, 2)
        # H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

class SegOverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W

class SegAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        print(q.shape)
        print(k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        print(attn.shape)
        print(v.shape)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Attention(nn.Module):
    def __init__(self, num_embeddings=64, sr_ratio=1, num_heads=1):
        super().__init__()        
               
        self.num_heads = num_heads
        self.num_embeddings = num_embeddings
        self.sr_ratio = sr_ratio
        self.scale = 1. / (sr_ratio*sr_ratio)
        #k_dim = num_embeddings // sr_ratio

        self.q = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, bias=False)

        self.k = nn.Conv2d(num_embeddings, num_embeddings, kernel_size=sr_ratio, stride=sr_ratio)
        self.kbn = nn.BatchNorm1d(num_embeddings)
        self.k2 = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, bias=False)

        # self.q2 = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, stride=1)
        # self.v = nn.AvgPool1d(dim)
        #self.rwmp = nn.MaxPool1d(k_dim)        
                
        self.conv = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1)

        
        #if sr_ratio > 1:
        #self.sr = nn.Conv2d(num_embeddings, num_embeddings, kernel_size=sr_ratio, stride=sr_ratio)            

    def forward(self, x, H, W):
        B, C, N = x.shape

        q_=self.q(x)
        q_ = q_.reshape(B, self.num_heads, C // self.num_heads, N)
        # print(q_.shape)
        # q_ = q_.permute(0,2,1,3)
        # print(q_.shape)
        # q_ = q_.transpose(2,1)

        k_ = x.reshape(B, C, H, W)
        k_ = self.k(k_)
        # k_ = k_.reshape(B, C, N//C)
        k_ = k_.reshape(B, C, -1)
        k_ = self.kbn(k_)
        k_ = self.k2(k_)        
        k_ = k_.reshape(B, self.num_heads, C // self.num_heads, N // (self.sr_ratio * self.sr_ratio))
        

        q_ = q_.transpose(-2, -1)
        print(q_.shape)
        print(k_.shape)
        
        attn =(q_ @ k_)
        attn = torch.mul(attn, self.scale)
        attn = torch.max(attn, 3)[0]        

        # attn = attn.reshape(B, C, N*self.sr_ratio)
        # attn = self.rwmp(attn)                        


        print(attn.shape)

        v = torch.mean(x, 2).expand(B, 1, self.num_embeddings)

        # v = torch.mean(x, 2).expand(B, 1, self.num_embeddings)
        # v = torch.tile(v, (B, self.num_embeddings, 1))

            
        # v = self.v(x).transpose(2,1)

        print(v.shape)

        attn = attn.transpose(2, 1)
        x = (attn @ v).transpose(2,1)

        x = self.conv(x)

        return x

class AttentionOrig(nn.Module):
    def __init__(self, dim, num_embeddings=3136, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()        

        self.dim = dim        
        k_dim = num_embeddings // (sr_ratio*sr_ratio)

        self.q = nn.Conv2d(num_embeddings, num_embeddings, kernel_size=8, stride=8, bias=qkv_bias)
        self.v = nn.AvgPool1d(num_embeddings)
        self.rwmp = nn.MaxPool1d(k_dim)
        self.k = nn.Conv1d(k_dim, k_dim, kernel_size=1)        
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        #if sr_ratio > 1:
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)            

    def forward(self, x, H, W):
        B, C, N = x.shape
        x_ = x.reshape(B, C, H, W)
        q_ = self.q(x_)
        q1_ = q_.reshape(B, C, N//C)
        
        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # x_ = x.reshape(B, C, H, W)
        k_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)            
            
        v = self.v(x.transpose(2,1)).transpose(1, 2)

        attn = (q_ @ k_.transpose(-2, -1)) 

        attn = self.rwmp(attn)                
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)                
        x = self.proj_drop(x)

        x = self.conv(x)

        return x


class MixFFN(nn.Module):
    def __init__(self, num_embeddings=64, hidden_features=256):
        super().__init__()        

        # notes
        # Is reshaping dimensions correct?
        # is First BN needed? Not shown in presentation but is in the paper
        # DW conv has padding is that correct
        # second reshape is different between Paper and presentation 

        self.hidden_features = hidden_features
        self.conv1 = nn.Conv1d(num_embeddings, hidden_features, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(hidden_features, num_embeddings, kernel_size=1)

    def forward(self, x, H, W):
        B, C, N = x.shape
        x = self.conv1(x)
        x = x.reshape(B, self.hidden_features, H, W)
        x = self.bn1(x)
        x = self.dwconv(x)
        x = x.reshape(B, self.hidden_features, N)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Block(nn.Module):
    def __init__(self, num_embeddings=64, hidden_features=256, sr_ratio=2):
        super().__init__()        
        self.attn = Attention(num_embeddings, sr_ratio=sr_ratio)
        self.ffn = MixFFN(num_embeddings=num_embeddings, hidden_features=hidden_features)

    def forward(self, x, H, W):
        x = x + self.attn(x, H, W)
        x = x + self.ffn(x, H, W)

        return x


class DepthModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dims=[64, 128, 250, 320],
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], in_chans=3, mlp_ratio=4):
        super().__init__()

        self.depths = depths
        self.embed_dims = embed_dims        

        self.patch_embed1 = PatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans, 
                                embed_dim=embed_dims[0])
        self.patch_embed2 = PatchEmbed(img_size=(img_size[0] // 4, img_size[1] //4), patch_size=3, stride=2, in_chans=embed_dims[0], 
                                embed_dim=embed_dims[1])
        self.patch_embed3 = PatchEmbed(img_size=(img_size[0] // 8, img_size[1] //8), patch_size=3, stride=2, in_chans=embed_dims[1], 
                                embed_dim=embed_dims[2])
        self.patch_embed4 = PatchEmbed(img_size=(img_size[0] // 16, img_size[1] //16), patch_size=3, stride=2, in_chans=embed_dims[2], 
                                embed_dim=embed_dims[3])


        self.block1 = nn.ModuleList([Block(
                num_embeddings=embed_dims[0], sr_ratio=sr_ratios[0], hidden_features=embed_dims[0] * mlp_ratio)
                for j in range(depths[0])])                    

        self.block2 = nn.ModuleList([Block(
                num_embeddings=embed_dims[1], sr_ratio=sr_ratios[1], hidden_features=embed_dims[1] * mlp_ratio)
                for j in range(depths[1])])                    

        self.block3 = nn.ModuleList([Block(
                num_embeddings=embed_dims[2], sr_ratio=sr_ratios[2], hidden_features=embed_dims[2] * mlp_ratio)
                for j in range(depths[2])])                    

        self.block4 = nn.ModuleList([Block(
                num_embeddings=embed_dims[3], sr_ratio=sr_ratios[3], hidden_features=embed_dims[3] * mlp_ratio)
                for j in range(depths[3])])                    

        
    def forward(self, x):
        outs = []
        B = x.shape[0]

        #stage 1
        x, (H, W) = self.patch_embed1(x)

        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #stage 2
        x, (H, W) = self.patch_embed2(x)

        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #stage 3
        x, (H, W) = self.patch_embed3(x)

        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        #stage 4
        x, (H, W) = self.patch_embed4(x)

        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)

        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)
        return outs
