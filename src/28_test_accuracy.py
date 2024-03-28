import torch

# 2分类问题： 0 1两类
outputs = torch.tensor([
    [0.1, 0.3],
    [0.8, 0.4]
])
print(outputs.argmax(1)) # 0-纵向 1横向 为一组数据 以下标为1类

predicts = outputs.argmax(1)
targets = torch.tensor([1, 1])
print(predicts == targets) # tensor([False,  True])
print((predicts == targets).sum().item())

# 数据准确率
data_size = len(predicts)
accuracy = (predicts == targets).sum().item() / data_size
print("准确率为：{:.2f}%".format(accuracy * 100.0))