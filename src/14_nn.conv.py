import torch.nn.functional as F
import torch

input = torch.tensor([ # input torch.Size([5, 5])
    [1, 2, 0, 3, 1],
    [0, 1, 2, 3, 1],
    [1, 2, 1, 0, 0],
    [5, 2, 3, 1, 1],
    [2, 1, 0, 1, 1]
])

kernel = torch.tensor([ # weight torch.Size([3, 3])
    [1, 2, 1],
    [0, 1, 0],
    [2, 1, 0]
])

input = torch.reshape(input, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

output = F.conv2d(input, kernel, None, stride=1)
output2 = F.conv2d(input, kernel,None, stride=1,padding=1)
print(output2)
