import torch
from torch import nn
import math
class eca_block(nn.Module):
    # 对于eca来说，我们要计算卷积核大小，所以我们要输入通道数，通过通道数计算卷积核的大小
    def __init__(self, channel, gamma = 2, b =1):
        super(eca_block, self).__init__()
        # 卷积核大小计算的公式,根据通道数，自适应计算卷积核的大小
        kernel_size = int(abs(math.log(channel, 2)+b / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size +1
        # 在卷积前要设置padding的值
        padding = kernel_size //2

        # 先对输入的内容进行平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #定义1d卷积
        self.conv = nn.Conv1d(1, 1, kernel_size, padding= padding, bias = False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        b, c, h, w = x.size()
        #平均池化的结果reshape一下，调成一个序列的形式
        avg = self.avg_pool(x).view([b, 1, c]) #第一个特征batchsize,第二胎个维度代表每个step代表特征长度，第三个维度代表每个时序
        out  =self.conv(avg)
        # 对sigmoid后的结果再进行一个reshape,方便后面的处理
        out = self.sigmoid(avg).view([b, c,1,1])
        print(out)
        return out *x
model = eca_block(512)
print(model)
inputs = torch.ones([2, 512, 26, 26])
putputs = model(inputs)


