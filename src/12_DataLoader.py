import torchvision.datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./torchdata", train=False, transform=torchvision.transforms.ToTensor(),download=True)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试数据集中第一张图片及其label
print("# 测试数据集中第一张图片及其label")
img, label = test_data[0]
print(img.shape)
print(label)

writer = SummaryWriter("../logs")

step = 0
for data in test_loader:
    imgs, labels = data
    # print("imgsize",img.shape) # torch.Size([4, 3, 32, 32])
    # print(label.shape)
    writer.add_images("test_data_drop_last", imgs, step)
    step += 1

writer.close()