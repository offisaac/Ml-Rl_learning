from click.core import F
from torch import nn
class my_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.convi1=nn.Conv2d(1,20,5)#输入图像的通道数为1 输出20个特征图(卷积核） 卷积和的大小是5x5像素
        self.convi2=nn.Conv2d(20,20,5)#这里的输入通道和输出通道相同 可能和神经网络有关？

    def forward(self,input):
        input=F.relu(self.convi1(input))
        return F.relu(self.convi2(input))

