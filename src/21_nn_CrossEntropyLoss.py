import torch
from torch import nn

input = torch.tensor([0.1, 0.2, 0.3])
target = torch.tensor([1])
input = torch.reshape(input, (1,3)) # 1batch 3classes
loss = nn.CrossEntropyLoss()
result = loss(input, target)
print(result) # tensor(1.1019)