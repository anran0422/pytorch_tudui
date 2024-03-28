import torchvision
from torch.nn import Linear

vgg16 = torchvision.models.vgg16(progress=True)

vgg16.classifier.add_module("add_Linear", Linear(1000,10))
print(vgg16)