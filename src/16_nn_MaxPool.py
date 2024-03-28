import torch.nn as nn
import torchvision.datasets
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("torchdata", train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

# 搭建简单神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
simpleNet = Net()

writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, labels = data
    writer.add_images("input",imgs,step)
    output = simpleNet(imgs)
    writer.add_images("output",output,step)
    step += 1
writer.close()