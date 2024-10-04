import torch
from torch.utils.data import Dataset
from PIL import Image
import os #python中关于操作系统的库 可以实现和系统的交互
# print(torch.__version__)
# print(torch.cuda.is_available())
# print(torch.randn((2,3)))

class  MyData(Dataset):
    # def __init__(self,root_dir,label_dir):#这里输入1.外层文件夹地址 2.内层文件夹名称(label)#这就是形象的处理方法
    #     self.root_dir = root_dir
    #     self.label_dir = label_dir
    #     self.path = os.path.join(self.root_dir,self.label_dir)#得到内层文件夹路径
    #     self.img_name_list =os.listdir(self.path)#得到内层文件夹条目列表
    def __init__(self, path):#传入目标文件夹地址
        self.path = path
        self.img_name_list=os.listdir(path)

    def __getitem__(self,index):
        img_name=self.img_name_list[index]
        img_item_path=os.path.join(self.path,img_name)
        img=Image.open(img_item_path)#通过路径返回图片类型对象(包含图片的各种特诊)
        label=self.img_name_list[index]
        return img,label

    def __len__(self):#长度返回目标文件夹的条目数目
        return len(self.img_name_list)

#总结 init函数输入的是外层文件夹地址和内层文件/文件夹名称 内部还得到内层文件夹内部的图片名称列表(通过os.path.join()进入内层文件夹 再"读取"得到内层条目名称)
# getitem函数输入的是index 返回的是内层文件夹中第index个图片和名称 方式是index读取路径列表得到对应名称 使用os.path.join()方法得到路径 使用open函数打开路径返回图片对象

#更改后的代码思路 传入目标文件夹路径 得到文件夹内部条目名称列表 index读取对应名字 和目标文件夹路径一同合成图片文件夹路径 通过open函数返回图片对象
img_path = r"E:\code for py\pytorch_learning_note\dataset\cat\cat-4756360_640.jpg"
img = Image.open(img_path)
# img.show(command='start')

cat_dataset=MyData(r"E:\code for py\pytorch_learning_note\dataset\cat")
print(len(cat_dataset))
print(cat_dataset[1][1])#返回第二张图片的名称
print(cat_dataset[1][0])#返回第二章图片对象
print(cat_dataset[2])

dog_dataset=MyData(r"E:\code for py\pytorch_learning_note\dataset\dog")

animal_dataset=cat_dataset+dog_dataset#相当于叠合了 dog数据集index紧跟着cat
