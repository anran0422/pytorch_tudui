from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter('../logs') # 日志event存放
img_path = "../dataset/train/ants/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)
print(type(img_array)) # <class 'numpy.ndarray'>
print(img_array.shape) # (512, 768, 3)

"""
    tag:名称
    img_tensor: 图片类型
    global_step: 训练次数
    dataformats: CHW HWC
"""
writer.add_image('test', img_array, global_step=3, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y = 2x", 2*i, i)

writer.close()