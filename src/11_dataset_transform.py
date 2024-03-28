import torchvision
from torch.utils.tensorboard import SummaryWriter

# compose组合多种功能
dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./torchdata", transform=dataset_transform, train=True, download=True)
test_set = torchvision.datasets.CIFAR10(root="./torchdata", transform=dataset_transform, train=False, download=True)

writer = SummaryWriter(log_dir="../logs")
for i in range(10):
    img, label = train_set[i]
    writer.add_image("test_set", img, i)

writer.close()