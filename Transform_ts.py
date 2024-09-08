from  torchvision.transfortensorboardms import transforms
from torch.utils. import SummaryWriter
from PIL import Image
import matplotlib.pyplot as plt
import  torchvision

image_path=r"E:\code for py\pytorch_learning_note\dataset\cat\cat-4756360_640.jpg"
image=Image.open(image_path)
image_tensor=transforms.ToTensor()(image)
with SummaryWriter("Tf_logs") as writer:
    writer.add_image("1", image_tensor, 1, dataformats='CHW')
# image = Image.open(image_path)
# image=transforms.Resize((255,255))(image)
# image_tensor=transforms.ToTensor()(image)
# with SummaryWriter("Tf_logs") as writer:
#     writer.add_image("1", image_tensor, 3, dataformats='CHW')
# image = Image.open(image_path)
# image_tensor=transforms.ToTensor()(image)
# image_tensor=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image_tensor)
# with SummaryWriter("Tf_logs") as writer:
#     writer.add_image("1", image_tensor, 5)

train_dataset=torchvision.datasets.CIFAR10(root="./dataset",train=True,download=True)

img,target=train_dataset[0]#数据集不是一个列表 这个数据集有内部自己定义的__getitem__方法 数据集本身是一个类 其内存在很多对象 这里内部调用data里的参数

print(img)
print(target)
img.show()