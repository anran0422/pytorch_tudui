import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='torchdata',train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1 = nn.Linear(196608, 10, bias=True)

    def forward(self,input):
        output = self.linear1(input)
        return output
simpleNet = Net()

for data in dataloader:
    imgs, lables = data
    print("训练前",imgs.shape)
    net_input =  torch.flatten(imgs)
    print("渴望输入尺寸",net_input.shape)
    output = simpleNet(net_input)
    print("训练完",output.shape)