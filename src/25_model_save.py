import torch
import torchvision
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16()

# 保存方式 1
# torch.save(vgg16, '../model/vgg16_method1.pth')

# 保存方式 2：模型参数（官方推荐）
# torch.save(vgg16.state_dict(), '../model/vgg16_method2.pth')

# 陷阱1
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv2d(3,64,3)

    def forward(self, x):
        x = self.conv1(x)
        return x
simpleNet = Net()
torch.save(simpleNet, '../model/simpleNet.pth')