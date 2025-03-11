import torch
from torch import nn

class senet(nn.Module):
    #因为要考虑输入进来的通道数，所以要传入，还要传入一个ratio,代表缩放的比例，第一次连接的比例少
    def __init__(self, channel, ratio=16):
        super(senet, self).__init__()   # 初始化的一个过程
        # 在高宽上进行平均池化，在完成平均池化后高宽是1了，所以自适应平均池化的参数设为1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 定义两次全连接
        self.fc = nn.Sequential(
            # 定义一个神经元个数较少的全连接
            nn.Linear(channel, channel//ratio, False),  #不使用偏置量，设为False
            nn.ReLU(),  #激活函数
            nn.Linear(channel // ratio, channel,False),
            nn.Sigmoid(),
        )
    def forward(self,x):
        # 特征层的size，第一维度是batchsize,第二维度是通道数，第三维度是高，第四维度是宽
        b, c, h, w = x.size()
        # b, c, h, w -> b, c, 1, 1(平均池化后的形状是这样的,我们要去掉后两个维度，所以要reshape一下，用view)
        avg = self.avg_pool(x).view([b, c])
        #全连接层后，把它的宽高维度添加上去，再reshape下
        # b, c -> b, c//ratio ->b,c -> b, c, 1, 1
        fc = self.fc(avg).view([b, c, 1, 1])
        # 输出每个通道的权值
        print(fc)
        #之后将两次全连接后的结果乘上输入进来的特征层
        return x * fc

 # 输入通道数是512
model = senet(512)
print(model)
#随便定义一个输入看看
inputs = torch.ones([2, 512, 26, 26])

outputs = model(inputs)

