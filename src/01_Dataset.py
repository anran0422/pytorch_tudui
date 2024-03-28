from torch.utils.data import Dataset
from PIL import Image
import os

class myDataset(Dataset):
    def __init__(self, root_path, label_path):
        self.root_path = root_path
        self.label_path = label_path
        self.path = os.path.join(self.root_path, self.label_path)

        self.current_image_file = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.current_image_file[index]
        img_item_path = os.path.join(self.root_path, self.label_path, img_name)
        img = Image.open(img_item_path)
        label = self.label_path
        return  img, label

    def __len__(self):
        return len(self.current_image_file)

root_path = "../dataset/train"
ants_label_path = "ants"
bees_label_path = "bees"
ants_dataset = myDataset(root_path, ants_label_path)
bees_dataset = myDataset(root_path, bees_label_path)
# # 拼接数据集
# train_dataset = ants_dataset + bees_dataset

# 将图片的label值存放在另外一个文件夹中，以图片名称为文件名称，其中存储图片label
root_dir = "../dataset/train"
target_dir = "ants"
img_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir
out_dir = "ants_label"
for i in img_path:
    file_name = i.split('.jpg')[0]
    with open(os.path.join(root_path, out_dir, "{}.txt".format(file_name)), 'w') as f:
        f.write(label)