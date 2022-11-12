from turtle import forward
import torch
import torch.nn as nn
from reducer import Reducer
from einops.layers.torch import Reduce

class SelfAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction_ratio):
        super().__init__()
        self.rr = reduction_ratio

        c = int(in_channels * in_channels * reduction_ratio)
        self.convq = nn.Conv1d(in_channels=c, out_channels=c, kernel_size=1)

        rc = int(in_channels * in_channels / reduction_ratio)
        self.convk = nn.Conv1d(in_channels=rc, out_channels=rc, kernel_size=1)
        self.avgpoolv = nn.AvgPool1d(kernel_size=c, stride=c)
        self.maxpool = nn.MaxPool1d(kernel_size=in_channels, stride=in_channels)
        # self.avgpoolv = Reduce('b c h w -> b 1 h w', 'mean')
        # self.reducer = Reducer(reduction_ratio=reduction_ratio, channels=in_channels)        
        self.reducer = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=reduction_ratio, stride=reduction_ratio)
        
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
        print("x1: " + str(x_.shape))
        x_ = self.reducer(x_)
        print("x2: " + str(x_.shape))
        x_ = x_.reshape(B, C, -1)
        print("x3: " + str(x_.shape))
        x_ = x_.permute(0, 2, 1)
        print("x4: " + str(x_.shape))
        

        k = self.convk(x_)
        print("k: " + str(k.shape))

        ktemp = k.transpose(-2, -1)
        print("ktemp: " + str(ktemp.shape))

        kv = k.reshape(B, -1, 2, 1, C)
        print("kv: " + str(kv.shape))

        kv = kv.permute(2, 0, 3, 1, 4)
        print("kv2: " + str(kv.shape))

        q = self.convq(x)
        print("q: " + str(q.shape))

        attn = q  @ k.transpose(-2, -1) 
        # attn = torch.matmul(q, k)
        print("nv: " + str(attn.shape))

        kq = self.maxpool(attn)
        print("kq: " + str(kq.shape))


        v = self.avgpoolv(x.permute(0, 2, 1)).permute(0, 2, 1)

        print("v: " + str(v.shape))

        out = kq @ v.transpose(-2, -1)

        # red = self.reducer(x)
        # print("reducer: " + str(red.shape))

        # q = self.convq(x)
        # k = self.convk(red)
        # v = self.avgpoolv(x)

        
        
        # print("q: " + str(q.shape))
        # print("k: " + str(k.shape))
        # print("v: " + str(v.shape))


    
    

