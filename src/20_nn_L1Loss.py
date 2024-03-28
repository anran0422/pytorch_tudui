import torch
from torch import nn
from torch.nn import L1Loss

input = torch.tensor([1,2,3], dtype=torch.float32)
target = torch.tensor([1,2,5], dtype=torch.float32)

# 这个处理，可能是为了符合 N C H W的书写规范
input = torch.reshape(input,(1,1,1,3))
target = torch.reshape(target, (1,1,1,3))

loss1 = L1Loss(reduction='sum')
result = loss1(input, target)
print("sum求loss", result)

loss_mse = nn.MSELoss(reduction='mean')
result = loss_mse(input, target)
print("mse方差计算loss",result)