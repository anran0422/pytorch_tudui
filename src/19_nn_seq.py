import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Sequential, Linear
from torch.utils.tensorboard import SummaryWriter

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model1 = Sequential(
            Conv2d(3,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,32,5,1,2),
            MaxPool2d(2),
            Conv2d(32,64,5,1,2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self,x):
        x = self.model1(x)
        return x

simpleNet = Net()
print(simpleNet)

input = torch.ones((64,3,32,32))
output = simpleNet(input)
print(output.shape)

writer = SummaryWriter('../logs')
writer.add_graph(simpleNet,input)
writer.close()