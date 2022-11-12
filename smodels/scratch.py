
import torch
import torch.nn as nn

x = torch.randn((1, 4, 4))


l = nn.LayerNorm(4)
c = nn.MaxPool1d(4) 
d = nn.AvgPool1d(4)

#c = nn.Conv1d(4, 4, 1, 1)

l_ = l(x)

print(l_)
print(l_.shape)

x_ = c(x)


d1_ = x.transpose(2,1)
d_ = d(d1_)
d2_ = d_.transpose(1,2)

print(x)
# print(x_)
# print(x_.shape)


print(d1_)
print(d1_.shape)

print(d_)
print(d_.shape)

print(d2_)
print(d2_.shape)
