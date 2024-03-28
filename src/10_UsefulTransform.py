from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("../logs")
img = Image.open("image/4.png").convert("RGB")
print(img)

# Totensor
tensor_obj = transforms.ToTensor()
tensor_img = tensor_obj(img)
writer.add_image("pytorch", tensor_img, 1)


# Nomalize 归一化
print(tensor_img[0][0][0]) # tensor(0.9922)
tans_norm_obj = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
norm_img = tans_norm_obj(tensor_img)
print(norm_img[0][0][0]) # 0层 0行 0列
writer.add_image("Normlize", norm_img, 1) # tensor(0.9843)

# Resize
print(img.size)
trans_resize = transforms.Resize((512,512))
# PIL -> PIL resize
resize_img = trans_resize(img)
# PIL resize -> Tensor resize
resize_img_totensor = tensor_obj(resize_img)
print(resize_img.size)
print(resize_img_totensor)
writer.add_image("Resize", resize_img_totensor, 0)

# Compose
trans_resize_2 = transforms.Resize(512) # Resize
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_obj])
img_resize_2 = trans_compose(img) # Tensor
writer.add_image("Resize", img_resize_2, 1)

# RandomCrop
trans_random = transforms.RandomCrop(256)
trans_compose_2 = transforms.Compose([trans_random, tensor_obj]) # 将PIL-> Tensor
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("Random Crop", img_crop, i)

writer.close()