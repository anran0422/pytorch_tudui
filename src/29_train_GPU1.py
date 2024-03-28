import torch.optim
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time


# 准备数据集
train_data = torchvision.datasets.CIFAR10('torchdata/', train=True, download=True,
                                              transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10('torchdata/', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集长度：{}".format(train_data_size))
print("测试数据集长度：{}".format(test_data_size))

# DataLoader加载数据集
train_data_loader = DataLoader(train_data, batch_size=64)
test_data_loader = DataLoader(test_data, batch_size=64)

# 搭建神经网络
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
if torch.cuda.is_available():
    simpleNet = simpleNet.cuda() # 网络模型转到cuda

# 损失函数-CrossEntropyLoss
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

# 优化器-SGD
learning_rate = 0.01
optimizer = torch.optim.SGD(simpleNet.parameters(), lr=learning_rate)

# 设置训练网络的参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 轮次
epoch = 10

# tensorboard记录
writer = SummaryWriter("../logs")

start_time = time.time()
for i in range(epoch):
    print("-----第{}轮训练开始-----".format(i + 1))

    # 训练步骤开始
    simpleNet.train(True)
    for data in train_data_loader:
        imgs, targets = data
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        outputs = simpleNet(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}, Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    simpleNet.eval()
    # 每个轮次整个测试数据集的loss
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_data_loader:
            imgs,targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = simpleNet(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss.item()
            accuracy = outputs.argmax(dim=1).eq(targets).sum().item()
            total_accuracy += accuracy

    total_test_step += 1
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{:.2f}%".format( total_accuracy / test_data_size * 100.0))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy, total_test_step)

    # 保存每个轮次的模型
    # torch.save(simpleNet, "simpleNet_{}.pth".format(i + 1))
    torch.save(simpleNet.state_dict(), "simpleNet_{}.pth".format(i+1))
    print("模型已保存")

writer.close()