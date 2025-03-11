import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from model import ZFNet
import os
import json
import time
from torch.optim.lr_scheduler import StepLR  # 引入学习率调度器

# device : GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
# 数据转换
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

# data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
data_root = os.getcwd()
image_path = data_root + "/cardata/"  # flower data set path
train_dataset = datasets.ImageFolder(root=image_path + "/train",
                                     transform=data_transform["train"])
train_num = len(train_dataset)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx
cla_dict = dict((val, key) for key, val in flower_list.items())
# write dict into json file

json_str = json.dumps(cla_dict, indent=3)
with open('class_indices.json', 'w') as json_file:
    json_file.write(json_str)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)

validate_dataset = datasets.ImageFolder(root=image_path + "/val",
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=0)

test_data_iter = iter(validate_loader)
test_image, test_label = next(test_data_iter)

net = ZFNet(num_classes=4, init_weights=True)

# 初始化优化器
optimizer = optim.Adam(net.parameters(), lr=0.0002)

# 初始化学习率调度器，每1个epoch后将学习率乘以0.1
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

net.to(device)
# 损失函数:这里用交叉熵
loss_function = nn.CrossEntropyLoss()
# 优化器 这里用Adam
optimizer = optim.Adam(net.parameters(), lr=0.0002)
# 训练参数保存路径
save_path = './AlexNet.pth'
# 训练过程中最高准确率
best_acc = 0.0

# 记录训练开始时间
start_time = time.perf_counter()

for epoch in range(5):
    # train
    net.train()  # 训练过程中，使用之前定义网络中的dropout
    running_loss = 0.0
    correct_predictions = 0  # 新增：用于计算训练准确率
    total_predictions = 0  # 新增：用于计算训练准确率
    t1 = time.perf_counter()
    for step, data in enumerate(train_loader, start=0):
        images, labels = data
        optimizer.zero_grad()
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # 更新训练损失
        running_loss += loss.item()

        # 计算训练准确率
        _, predicted = torch.max(outputs, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels.to(device)).sum().item()

        # 打印训练进度（这部分可以保持不变）
        rate = (step + 1) / len(train_loader)
        a = "*" * int(rate * 50)
        b = "." * int((1 - rate) * 50)
        print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        # 计算平均训练损失和训练准确率
    train_loss = running_loss / step
    train_accuracy = correct_predictions / total_predictions
    print()  # 换行
    print(time.perf_counter() - t1)  # 打印训练时间
    print(f'Train Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')  # 新增：打印训练损失和准确率

    # validate
    net.eval()  # 测试过程中不需要dropout，使用所有的神经元
    val_running_loss = 0.0  # 新增：用于计算测试损失
    val_correct_predictions = 0  # 新增：用于计算测试准确率
    val_total_predictions = 0  # 新增：用于计算测试准确率
    with torch.no_grad():
        for val_data in validate_loader:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            loss = loss_function(outputs, val_labels.to(device))  # 计算测试损失
            val_running_loss += loss.item()

            # 计算测试准确率
            _, val_predicted = torch.max(outputs, 1)
            val_total_predictions += val_labels.size(0)
            val_correct_predictions += (val_predicted == val_labels.to(device)).sum().item()

            # 计算平均测试损失和测试准确率
    test_loss = val_running_loss / len(validate_loader)
    test_accuracy = val_correct_predictions / val_num  # 确保val_num已经定义且正确

    # 保存最佳模型（这部分可以保持不变）
    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(net.state_dict(), save_path)

        # 将每个epoch的结果写入output.txt文件
    with open('output.txt', 'a', encoding='utf-8') as file:  # 使用'a'模式以追加内容
        file.write(
            f'[epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}\n')

    # 训练结束后计算总耗时
total_time_taken = time.perf_counter() - start_time

# 将总耗时写入output.txt文件
with open('output.txt', 'a', encoding='utf-8') as file:
    file.write(f'Total time taken for training: {total_time_taken:.2f} seconds\n')

print('Finished Training')  # 训练结束

# # 开始进行训练和测试，训练一轮，测试一轮
# for epoch in range(5):
#     # train
#     net.train()    # 训练过程中，使用之前定义网络中的dropout
#     running_loss = 0.0
#     t1 = time.perf_counter()
#     for step, data in enumerate(train_loader, start=0):
#         images, labels = data
#         optimizer.zero_grad()
#         outputs = net(images.to(device))
#         loss = loss_function(outputs, labels.to(device))
#         loss.backward()
#         optimizer.step()
#
#         # print statistics
#         running_loss += loss.item()
#         # print train process
#         rate = (step + 1) / len(train_loader)
#         a = "*" * int(rate * 50)
#         b = "." * int((1 - rate) * 50)
#         print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
#     print()
#     print(time.perf_counter()-t1)
#
#     # validate
#     net.eval()    # 测试过程中不需要dropout，使用所有的神经元
#     acc = 0.0  # accumulate accurate number / epoch
#     with torch.no_grad():
#         for val_data in validate_loader:
#             val_images, val_labels = val_data
#             outputs = net(val_images.to(device))
#             predict_y = torch.max(outputs, dim=1)[1]
#             acc += (predict_y == val_labels.to(device)).sum().item()
#         val_accurate = acc / val_num
#         if val_accurate > best_acc:
#             best_acc = val_accurate
#             torch.save(net.state_dict(), save_path)
#         print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
#               (epoch + 1, running_loss / step, val_accurate))
#
# print('Finished Training')


# # 开始进行训练和测试，训练一轮，测试一轮
# for epoch in range(10):
#     # train
#     net.train()  # 训练过程中，使用之前定义网络中的dropout
#     running_loss = 0.0
#     correct_predictions = 0  # 新增：用于计算训练准确率
#     total_predictions = 0  # 新增：用于计算训练准确率
#     t1 = time.perf_counter()
#     for step, data in enumerate(train_loader, start=0):
#         images, labels = data
#         optimizer.zero_grad()
#         outputs = net(images.to(device))
#         loss = loss_function(outputs, labels.to(device))
#         loss.backward()
#         optimizer.step()
#
#         # 更新训练损失
#         running_loss += loss.item()
#
#         # 计算训练准确率
#         _, predicted = torch.max(outputs, 1)
#         total_predictions += labels.size(0)
#         correct_predictions += (predicted == labels.to(device)).sum().item()
#
#         # 打印训练进度（这部分可以保持不变）
#         rate = (step + 1) / len(train_loader)
#         a = "*" * int(rate * 50)
#         b = "." * int((1 - rate) * 50)
#         print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
#
#         # 计算平均训练损失和训练准确率
#     train_loss = running_loss / step
#     train_accuracy = correct_predictions / total_predictions
#     print()  # 换行
#     print(time.perf_counter() - t1)  # 打印训练时间
#     print(f'Train Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')  # 新增：打印训练损失和准确率
#
#     # validate
#     net.eval()  # 测试过程中不需要dropout，使用所有的神经元
#     val_running_loss = 0.0  # 新增：用于计算测试损失
#     val_correct_predictions = 0  # 新增：用于计算测试准确率
#     val_total_predictions = 0  # 新增：用于计算测试准确率
#     with torch.no_grad():
#         for val_data in validate_loader:
#             val_images, val_labels = val_data
#             outputs = net(val_images.to(device))
#             loss = loss_function(outputs, val_labels.to(device))  # 计算测试损失
#             val_running_loss += loss.item()
#
#             # 计算测试准确率
#             _, val_predicted = torch.max(outputs, 1)
#             val_total_predictions += val_labels.size(0)
#             val_correct_predictions += (val_predicted == val_labels.to(device)).sum().item()
#
#             # 计算平均测试损失和测试准确率
#     test_loss = val_running_loss / len(validate_loader)
#     test_accuracy = val_correct_predictions / val_num
#
#     # 保存最佳模型（这部分可以保持不变）
#     if test_accuracy > best_acc:
#         best_acc = test_accuracy
#         torch.save(net.state_dict(), save_path)
#
#         # 将每个epoch的结果写入output.txt文件
#         with open('output.txt', 'a', encoding='utf-8') as file:
#             file.write(
#                 f'[epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, '
#                 f'test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}\n')
#
#         print(
#             f'[epoch {epoch + 1}] train_loss: {train_loss:.3f}, train_accuracy: {train_accuracy:.3f}, '
#             f'test_loss: {test_loss:.3f}, test_accuracy: {test_accuracy:.3f}')
#
# print('Finished Training')  # 训练结束
