import torch
import torchvision
from torch.nn import Conv2d

# 方式1
# model = torch.load("../model/vgg16_method1.pth")
# print(model)


# 方式2
# model = torch.load("model/vgg16_method2.pth")
# vgg16 = torchvision.models.vgg16()
# vgg16.load_state_dict(torch.load("../model/vgg16_method2.pth"))
# print(vgg16)



# 方式1：陷阱
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x

model = torch.load("simpleNet.pth") # simpleNet = Net()
print(model)