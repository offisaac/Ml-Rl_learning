import torchvision
import torchvision.transforms as transforms
import numpy as np

import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataloader

import torch
import torch.optim as optim
import torch.nn as nn

import time

compose_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])
])
to_tensor = transforms.ToTensor()
train_data = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=compose_transform)
test_data = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=compose_transform)
print(train_data.data[100][11][11])

# class mydataset(dataset.Dataset):
#     def __init__(self, dataset):
#         # 初始化变量
#         self.idx_to_class = {}
#         self.Y_unique_heating_code = []
#
#         # 构建类索引映射
#         for key in dataset.class_to_idx:
#             self.idx_to_class[dataset.class_to_idx[key]] = key
#
#         # 产生独热编码
#         unique_heating_code = torch.eye(10)  # 独热编码矩阵
#         for i in dataset.targets:
#             self.Y_unique_heating_code.append(unique_heating_code[i])  # 创建真实值对应的独热编码
#         self.Y_unique_heating_code = torch.stack(self.Y_unique_heating_code)  # 转换为张量
#
#         # 获取数据
#         self.X = dataset.data
#         self.class_num = len(dataset.class_to_idx)
#
#     def __getitem__(self, index):
#         # 获取输入图像数据及其独热编码标签
#         return torch.tensor(self.X[index], dtype=torch.float32).permute(2, 0, 1), self.Y_unique_heating_code[index]
#
#     def __len__(self):
#         return len(self.X)
class mydataset(dataset.Dataset):
    def __init__(self,dataset):
        self.idx_to_class={}
        self.Y_unique_heating_code=[]
        for key in dataset.class_to_idx:
            self.idx_to_class[dataset.class_to_idx[key]]=key#供最后一步独热编码取最大值对应使用
            unique_heating_code=torch.tensor(np.eye(10))#产生独热编码
            for i in dataset.targets:
                self.Y_unique_heating_code.append(unique_heating_code[:,i])#创建真实值对应独热编码
            self.X=dataset.data#data本身就是双array 对应元素是三通道 卷积层可以直接读取 这部分底层不需要人为操作
            self.Y = dataset.targets
            self.class_num=len(dataset.class_to_idx)
    def __getitem__(self,index):
        # return torch.tensor(self.X[index],dtype=torch.float32).permute(2,0,1),self.Y_unique_heating_code[index]
        return torch.tensor(self.X[index], dtype=torch.float32).permute(2, 0, 1), self.Y[
            index]  # 理论上应该使用独热编码 但是pytorch对数学公式做了处理 先得到输出的权重 再softmax得到概率 再取-log从取最大值到取最小值 再根据整数序列选取使用哪个输出概率(如果模型好 真实对应的就是最大的概率就是最小的log(P)输出
    def __len__(self):
        return len(self.X)

my_train_data=mydataset(train_data)
my_test_data=mydataset(test_data)
my_train_data_loader = torch.utils.data.DataLoader(my_train_data, batch_size=400, shuffle=True)
my_test_data_loader = torch.utils.data.DataLoader(my_test_data, batch_size=400, shuffle=True)
class CNNClassifier(nn.Module):
    def __init__(self,class_num):
        super().__init__()#!外部的输入不应该放在类名后(作为继承) 而是放在init后
        self.conv1 = nn.Conv2d(3, 128, 3, stride=1, padding=1)  # 这里padding=1是指每个边都向外扩一个"0"的像素 所以对于行 相当于扩了两个像素
        self.pool=nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(128, 256, 3, stride=1,
                               padding=1)  # 实际设计图片的卷积层时 通道输出图层数由filter决定 最后给线性层输入的列向量行数字需要根据卷积后的图片尺寸决定 这个有公式计算
        self.conv3 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        #卷积操作影响图层个数和图的尺寸 池化只影响图的尺寸 图最终尺寸只和初始图尺寸 卷积的stride padding有关 和通道无关 通道只和图层有关
        self.func1 = nn.Linear(64 * 2 * 2, 128)  # 怎么从32通道8*8array对应到32*8*8个列向量 底层不用考虑
        self.func2 = nn.Linear(128, 128)
        self.func3 = nn.Linear(128, class_num)
        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.4)#Dropout用于训练更新参数时随机的让神经元的输出置0 但实际使用和评估时不做改变 因为CNN参数相对于全连接不太多 所以一般给的这个值不大
        self.sigmoid=nn.Sigmoid()
    def forward(self,Input):
        x = self.conv1(Input)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.conv4(x)
        x = self.pool(x)
        x = self.dropout(x)  #主要是为了防止过拟合使用 防止模型对某些神经元过于依赖 增加泛化能力(希望部分集训练的参数对全集也适应)
        x = x.view(x.size(0),-1)#x = x.view(-1,32*8*8) 这里-1可以在任意位置 表示该位置由输入和另一个位置求得 这里两式等效 将 x 的形状从 [batch size, 32, 8, 8] 变为 [batch size, 2048]。
        x = self.func1(x)
        x = self.func2(x)
        x = self.func3(x)
        #x = self.relu(x) #不用加 使用crossentropy 计算Loss内部会对Outputs先进行softmax再计算
        return x
model = CNNClassifier(my_train_data.class_num)
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0)
# optimizer = optim.RMSprop(model.parameters(), lr=0.0005,momentum=0.2)
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                       weight_decay=0)  # 如果没有二阶beta参数 Adam和RMSprop可以认为同效果
# epochs = 500
# for epoch in range(epochs):
#     for X, Y in my_train_data_loader:
#         outputs = model(X)
#         loss = criterion(outputs, Y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     if (epoch + 1) % 2 == 0:
#         print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
#     if (epoch) % 10 == 0:
#         torch.save(model.state_dict(), f'model_parameter_set/model_parameter_{epoch}')
#     if loss.item() < 0.001:
#         break

loaded_model = CNNClassifier(class_num=10)
loaded_model.load_state_dict(torch.load(f'./model_parameter_set/model_parameter_20', weights_only=True))
loaded_model.eval()  # 设置为评估模式 用于关闭某些正则化操作

# 测试模型的准确率 自己的 需要修改模型__getitem__返回值
# count_all=0
# count_right=0
# with torch.no_grad():  # 禁用梯度计算，以减少内存占用
#     for data,label in my_test_data_loader:
#         count_all+=len(data)#orcount_all+=data.size(0) 张量0维度 也就是batch维度的大小
#         _, predict = torch.max(model(data), 1)
#         # _, predict = torch.max(loaded_model(data), 1)
#         _,real=torch.max(label,1)
#         for i,j in zip(predict,real):
#             if i==j:
#                 count_right+=1
# print(f"the possibility of rightness is{count_right/count_all}")
# 这个不需要 和训练模型配套
correct = 0
total = 0
with torch.no_grad():  # 禁用梯度计算，以减少内存占用
    for data, labels in my_test_data_loader:
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)  # 获取每个样本的最大概率的索引，即预测的类别 1代表从行取 也就是batch计算出的结果其实是行堆叠
        total += labels.size(0)  # 表示在张量第零维的长度 张量本身的shape返回其各个维度的长度！！！
        correct += (
                    predicted == labels).sum().item()  # ==是两个vector判断等 相等处赋True 不等处赋值False 返回相同大小vector .sum是将其内所有数相加 .item是将张量类型数转化成普通类型数
print(f"the possibility of rightness is{correct / total}")

# 保存模型参数 直接保存到指定地址即可
# 导入模型参数 需要创建模型 导入模型参数 模型进行eval处理
timestamp = time.strftime("%Y%m%d_%H%M%S")
torch.save(model.state_dict(), f'model_parameter_set/model_parameter_{timestamp} possibility{correct / total}')



'''
本次的一些语法点
1. __getitem__(self,index) 输入要有self和index
2. dataset.Dataset 真实使用的类不是dataset而是Dataset
3. 定义函数def 定义类class
4. 对于列表或者字典 类内外都需要事先初始化
5. 定义空列表后如果想添加元素 不能像字典一样调用即添加 而是必须append显式添加
6 .[:,i]代表选取第i列 [:i]代表选取0-i行(因为有,说明有行列区分 没有,必定指行)
7. 图像训练应该将RGB值归一化 将数据转为float 将数据格式转为CHW
8. 加载时应该使用weights_only=True
9. 训练的时候使用的是DataLoader 测试的时候也应该用DataLoader 否则会报错
10. 测试的流程 导入模型参数 使用Dataloder进行循环读取数据 X输入模型得到输出(tensor) 输出和Y（tensor）进行比较返回bool(tensor) 对tensor进行.sum()操作得到比较成功总数 再.item()将tensor转为数据 训练的总数则每一次循环加上batch大小即可(gpt比较的是输出的最大值位置 返回设置的也是序号)
11. 自己的测试流程是返回独热编码 输出是每个类比的采信值 但是没办法直接比较(因为采信值和1还是有差别) 还是要比较max
12. item()只能对数tensor操作
13. for i,j in first 意思是first返回二元类型变量 for i,j in zip(first,second) 这个才是每个循环从first second各选一个
14. 张量均有shape .size(dim_num)表示返回dim_num维度的size
'''










