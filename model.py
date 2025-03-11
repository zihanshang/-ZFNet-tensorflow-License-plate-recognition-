# import torch.nn as nn
# import torch
# # from CBAM import CBAM
# # from SENet import senet
# class ZFNet(nn.Module):
#     def __init__(self, num_classes=4, init_weights=False):
#         super(ZFNet, self).__init__()
#         self.features = nn.Sequential(  # 打包
#             nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),   # input[3, 224, 224]  output[48, 110, 110] 自动舍去小数点后
#             nn.ReLU(inplace=True),  # inplace 可以载入更大模型
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # output[48, 55, 55] kernel_num为原论文一半
#
#
#             nn.Conv2d(48, 128, kernel_size=5, stride=2),            # output[128, 26, 26]
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),       # output[128, 13, 13]
#
#             nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#
#
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
#             nn.ReLU(inplace=True),
#             nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
#             nn.ReLU(inplace=True),
#
#             nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             # 全连接
#             nn.Linear(128 * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()
#
#     def forward(self, x):
#         x = self.features(x)
#         x = torch.flatten(x, start_dim=1)  # 展平   或者view()
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 何教授方法
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)  # 正态分布赋值
#                 nn.init.constant_(m.bias, 0)

            # 示例用法
# model = ZFNet(num_classes=10, init_weights=True)

# from Triplet import TripletAttention
# class ZFNet(nn.Module):
#     def __init__(self, num_classes=4, init_weights=False):
#         super(ZFNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM(48),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(48, 128, kernel_size=5, stride=2),
#             nn.ReLU(inplace=True),
#             CBAM(128),
#             eca_block(128),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#
#             nn.Conv2d(128, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM(192),
#             # TripletAttention(192),
#             nn.Conv2d(192, 192, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(192, 128, kernel_size=3, padding=1),
#             nn.ReLU(inplace=True),
#             CBAM(128),
#             senet(128),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#         )
#         self.classifier = nn.Sequential(
#             nn.Dropout(p=0.5),
#             nn.Linear(128 * 6 * 6, 2048),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.5),
#             nn.Linear(2048, 2048),
#             nn.ReLU(inplace=True),
#             nn.Linear(2048, num_classes),
#         )
#         if init_weights:
#             self._initialize_weights()
import torch.nn as nn
import torch
from CBAM import CBAM
from ECAnet import eca_block
from SENet import senet
class ZFNet(nn.Module):
    def __init__(self, num_classes=4, init_weights=False):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=7, stride=2, padding=1),
            nn.ReLU(inplace=True),
            CBAM(48),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 128, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            CBAM(128),
            # eca_block(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # # 新增的卷积层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 例如，添加了一个256通道的3x3卷积
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(192),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(128),
            # senet(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        # 注意：需要调整分类器的输入尺寸
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            # 假设特征图大小在添加新层后仍然保持为6x6（这取决于你的具体实现和输入尺寸）
            # 如果特征图大小改变，这里需要相应调整
            nn.Linear(128 * 6 * 6, 2048),  # 如果特征图大小改变，这里需要调整
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        # 权重初始化（与原始ZFNet相同）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)

