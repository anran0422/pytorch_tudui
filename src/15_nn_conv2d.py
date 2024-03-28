import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./torchdata",train=False,transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset,batch_size=64)

# 搭建简单神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=0, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        return x
simpleNet = Net()

writer = SummaryWriter("../logs")
# 将数据集图像放到这个神经网络中
step = 0
for data in dataloader:
    imgs, labels = data
    output = simpleNet(imgs)
    print("一组图片对比")
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input_imgs",imgs, step)
    output = torch.reshape(output,(-1,3,30,30))
    writer.add_images("out_imgs",output, step)
    step += 1

writer.close()