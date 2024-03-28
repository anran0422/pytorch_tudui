import torch
import torchvision
from torch import nn
from torch.nn import Sequential, Conv2d, MaxPool2d, Flatten, Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(root='torchdata',train=False,download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=1)
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

loss = nn.CrossEntropyLoss()

simpleNet = Net()
for data in dataloader:
    imgs, labels = data
    outputs = simpleNet(imgs)
    print("训练后的图片，及其原本的分类标签")
    print(outputs)
    print(labels.shape)
    print("结果与实际的差距，即损失")
    result_loss = loss(outputs,labels)
    # result_loss.backward() # 计算梯度 反向传播
    print(result_loss)
