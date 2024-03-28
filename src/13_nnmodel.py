import torch
import torch.nn as nn

class nnModel(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self, input):
        output = input + 1
        return output

myModel = nnModel()
x = torch.tensor(1.0)
output = myModel(x)
print(output)