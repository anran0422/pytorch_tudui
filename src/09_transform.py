from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
# python用法 -> tensor类型
# 通过transforms.totensor去看两个问题

img_path = "../dataset/train/ants/0013035.jpg"
img = Image.open(img_path)
print(img) # <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x129C8D93110>

writer = SummaryWriter("../logs")

# 1. transforms如何使用（python）
trans_tensor_obj = transforms.ToTensor()
tensor_img = trans_tensor_obj(img)

writer.add_image("Tensor_img", tensor_img, global_step=0)
writer.close()
