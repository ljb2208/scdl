
from matplotlib.cbook import simple_linear_interpolation
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

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

class Attention(nn.Module):
    def __init__(self, dim, num_embeddings=64, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()        

        self.dim = dim        
        k_dim = num_embeddings // (sr_ratio*sr_ratio)

        self.q = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1)

        self.k = nn.Conv2d(num_embeddings, num_embeddings, kernel_size=sr_ratio, stride=sr_ratio, bias=qkv_bias)
        self.kbn = nn.BatchNorm1d(num_embeddings)
        self.k2 = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1)

        self.q2 = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, stride=1)
        self.v = nn.AvgPool1d(num_embeddings)
        self.rwmp = nn.MaxPool1d(k_dim)        
                
        self.conv = nn.Conv1d(num_embeddings, num_embeddings, kernel_size=1, bias=qkv_bias)

        self.sr_ratio = sr_ratio
        #if sr_ratio > 1:
        self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)            

    def forward(self, x, H, W):
        B, C, N = x.shape

        q_=self.q(x)
        q_ = q_.reshape(B, C, H, W)
        print(q_.shape)
        # q_ = q_.permute(0,2,1,3)
        # print(q_.shape)
        # q_ = q_.transpose(2,1)

        k_ = x.reshape(B, C, H, W)
        k_ = self.k(k_)
        # k_ = k_.reshape(B, C, N//C)
        k_ = k_.reshape(B, C, -1)
        k_ = self.kbn(k_)
        k_ = self.k2(k_)        
        k_ = k_.reshape(B, C, H//self.sr_ratio, W//self.sr_ratio)
        

        print(q_.shape)
        print(k_.shape)


        attn =(q_ @ k_.transpose(-2, -1))

        x_ = x.reshape(B, C, H, W)
        q_ = self.q(x_).reshape(B, C, N//C)
        q_ = self.bn(q_)
        q_ = self.q2(q_)        
        
        # x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        # x_ = x.reshape(B, C, H, W)
        k_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)            
            
        v = self.v(x.transpose(2,1)).transpose(1, 2)

        attn = (q_ @ k_.transpose(-2, -1)) 

        attn = self.rwmp(attn)                        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)                        

        x = self.conv(x)

        return x

class AttentionOrig(nn.Module):
    def __init__(self, dim, num_features=3136, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()        

        self.dim = dim        
        k_dim = num_features // (sr_ratio*sr_ratio)

        self.q = nn.Conv2d(num_features, num_features, kernel_size=8, stride=8, bias=qkv_bias)
        self.v = nn.AvgPool1d(num_features)
        self.rwmp = nn.MaxPool1d(k_dim)
        self.k = nn.Conv1d(k_dim, k_dim, kernel_size=1)        
        self.attn_drop = nn.Dropout(attn_drop)        
        self.proj_drop = nn.Dropout(proj_drop)
        self.conv = nn.Conv1d(num_features, num_features, kernel_size=1, bias=qkv_bias)

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
        x = x.reshape(B, self.hidden_features, H//4, W//4)
        x = self.bn1(x)
        x = self.dwconv(x)
        x = x.reshape(B, self.hidden_features, N)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)

        return x

class Block(nn.Module):
    def __init__(self, dim=64, num_features=3136, sr_ratio=2):
        super().__init__()        
        self.attn = Attention(dim, num_features=num_features, sr_ratio=sr_ratio)
        self.ffn = MixFFN(num_features=num_features)

    def forward(self, x, H, W):
        x = x + self.attn(x, H, W)
        x = x + self.ffn(x)

        return x


class DepthModel(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dims=[64, 128, 250, 320],
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1], in_chans=3):
        super().__init__()

        self.depths = depths
        self.embed_dims = embed_dims

        for i in range(len(depths)):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)), 
                patch_size=patch_size if i==0 else 2, 
                embed_dim=embed_dims[i],
                in_chans=in_chans if i==0 else embed_dims[i-1])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], sr_ratio=sr_ratios[i], num_features=patch_embed.H * patch_embed.W)
                for j in range(depths[i])])

            setattr(self, f"patch_embed{i + 1}", patch_embed)            
            setattr(self, f"block{i + 1}", block)

    def forward(self, x):
        outs = []
        B = x.shape[0]

        for i in range(len(self.depths)):
            patch_embed = getattr(self, f"patch_embed{i + 1}")            
            block = getattr(self, f"block{i + 1}")

            x, (H, W) = patch_embed(x)

            for blk in block:
                x = blk(x, H, W)         

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   
            outs.append(x)

        return outs
