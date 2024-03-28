import torchvision
from PIL import Image

import torch
from torch import nn


img_dog = "image/img.png"
img_air = "image/img_1.png"

img_dog = Image.open(img_dog).convert('RGB') # mode=RGB size=202x217
img_air = Image.open(img_air).convert('RGB')# 变为3通道，实际上RGB就是3通道

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((32,32)),
])
# 改变尺寸 PIL->tensor
img_dog = transform(img_dog)
img_air = transform(img_air)

print(img_dog.shape)
print(img_air.shape)
# 变成4维张量
img_dog = torch.reshape(img_dog,(1,3,32,32))
img_air = torch.reshape(img_air,(1,3,32,32))

# 引入网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32,5,1,2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,5,1,2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4,64),
            nn.Linear(64,10)
        )
    def forward(self, x):
        x = self.model(x)
        return x
simpleNet = Net()
# 加载已经训练好的模型
simpleNet.load_state_dict(torch.load("../model/simpleNet_50.pth")) # 训练了epoch=50的模型
print(simpleNet)

# 将图片输入然后测试
simpleNet.eval() # 节约性能
with torch.no_grad():
    output_dog = simpleNet(img_dog)
    output_air = simpleNet(img_air)

dog_label_number = output_dog.argmax(1).item() # 横向查看哪一个类别最大
air_label_number = output_air.argmax(1).item()

classic = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
print("dog分类结果：", classic[dog_label_number])
print("airplane分类结果：", classic[air_label_number])