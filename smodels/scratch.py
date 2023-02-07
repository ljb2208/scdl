
import torch
import torch.nn as nn

x = torch.randn((1, 3136, 64))
y = torch.randn((1, 784, 64))

y_ = y.transpose(-2, -1)
z = x @ y_


l = nn.LayerNorm(4)
c = nn.MaxPool1d(4) 
d = nn.AvgPool1d(64)

c = nn.Conv1d(3136, 3136, 1, 1) 



# sr = nn.Conv2d(16, 16, kernel_size=2, stride=2)

# B, N, C = x.shape
# x_ = x.permute(0, 2, 1).reshape(B, C, 4, 4)

# x_ = sr(x_).reshape(B, C, -1).permute(0, 2, 1)            

# x_ = sr(x)


# l_ = l(x)

# print(l_)
# print(l_.shape)

x_ = c(x)


# d1_ = x.transpose(2,1)
# d_ = d(d1_)
# d2_ = d_.transpose(1,2)

print(x)
print(x.shape)
print(x_)
print(x_.shape)

x_ = d(x_)

print("avgpool")
print(x_)
print(x_.shape)

# print(d1_)
# print(d1_.shape)

# print(d_)
# print(d_.shape)

# print(d2_)
# print(d2_.shape)
