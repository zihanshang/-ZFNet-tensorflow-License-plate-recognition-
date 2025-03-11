# 最初的
# import torch
# from PIL.ImagePalette import random
# import  random
# from model import ZFNet
# from torchvision import transforms
# import json
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
# 
# def load_images_from_folder(folder):
#     images = []
#     for filename in os.listdir(folder):
#         img2 = Image.open(os.path.join(folder, filename))
#         img2 = img2.resize((224, 224))  # 你可以根据需要调整图片大小
#         img2 = np.array(img2)
#         images.append(img2)
# 
#     images = np.array(images)
#     return images
# 
# 
# # 指定包含图片的文件夹路径
# folder_path = r"C:\Users\35008\Desktop\cardataset2\cars"
# 
# # 加载图片数据集
# image_dataset = load_images_from_folder(folder_path)
# 
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 
# data_transform = transforms.Compose(
#     [
#      transforms.ToTensor(),
#      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# for i in range(100):
#     # load image
#     ss=random.randint(1,200)
#     img = image_dataset[ss]
#     img = cv2.resize(img, (224, 224))
#     #img = Image.open('roses.jpg')     # 验证玫瑰花
#     plt.imshow(img)
#     # [N, C, H, W]
#     img = data_transform(img)
#     # expand batch dimension
#     img = torch.unsqueeze(img, dim=0)
# 
#     # read class_indict
#     try:
#         json_file = open('./class_indices.json', 'r')
#         class_indict = json.load(json_file)
#     except Exception as e:
#         print(e)
#         exit(-1)
# 
#     # create model
#     model = ZFNet(num_classes=4)
#     # load model weights
#     model_weight_path = "./AlexNet.pth"
#     model.load_state_dict(torch.load(model_weight_path))
#     model.eval()
#     with torch.no_grad():
#         # predict class
#         output = torch.squeeze(model(img))
#         predict = torch.softmax(output, dim=0)
#         predict_cla = torch.argmax(predict).numpy()
#     plt.text(200, -8, f'Predict: {class_indict[str(predict_cla)]}\nAccuracy: {predict[predict_cla].item():.4f}', fontsize=12,
#              color='red', bbox=dict(facecolor='yellow', alpha=0.5))
#     with open('input.txt', 'a', encoding='utf-8') as file:
#         file.write(f'Predict: {class_indict[str(predict_cla)]} Accuracy: {predict[predict_cla].item():.4f}\n')
# 
#     print(i,class_indict[str(predict_cla)],"accurate:" ,predict[predict_cla].item())
#     plt.show()
#     i=i+1


# 会一张一张弹跳的
import torch
import random
from model import ZFNet
from torchvision import transforms
import json
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# def load_images_from_folder(folder):
#     images = []
#     filenames = []  # 用于存储图片的文件名
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#         img = Image.open(img_path)
#         img = img.resize((224, 224))  # 你可以根据需要调整图片大小
#         img = np.array(img)
#         images.append(img)
#         filenames.append(filename)  # 保存图片文件名
#
#     images = np.array(images)
#     return images, filenames
#
#
# # 指定包含图片的文件夹路径
# folder_path = r"C:\Users\35008\Desktop\cars"
#
# # 加载图片数据集和文件名
# image_dataset, filenames = load_images_from_folder(folder_path)
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# data_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]
# )
#
# # 加载类索引映射
# try:
#     json_file = open('./class_indices.json', 'r')
#     class_indict = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# # 创建模型并加载预训练权重
# model = ZFNet(num_classes=4)
# model_weight_path = "./AlexNet.pth"
# model.load_state_dict(torch.load(model_weight_path))
# model.eval()
#
# with open('input.txt', 'a', encoding='utf-8') as file:
#     for i in range(6):
#         # 随机选择一张图片
#         ss = random.randint(0, len(image_dataset) - 1)
#         img = image_dataset[ss]
#         img_filename = filenames[ss]  # 获取图片的文件名
#
#         # 假设文件名就是类别信息（例如：'cat_001.jpg'）
#         true_label = img_filename.split('_')[0]  # 获取文件名的前缀作为真实标签
#
#         img = cv2.resize(img, (224, 224))
#
#         # 显示图片
#         plt.imshow(img)
#
#         # 对图像进行预处理
#         img_tensor = data_transform(img)
#         img_tensor = torch.unsqueeze(img_tensor, dim=0)
#
#         # 进行预测
#         with torch.no_grad():
#             output = torch.squeeze(model(img_tensor))
#             predict = torch.softmax(output, dim=0)
#             predict_cla = torch.argmax(predict).numpy()
#
#         # 获取预测的类别名称和准确率
#         predicted_class = class_indict[str(predict_cla)]  # 使用类别名称而不是索引
#         accuracy = predict[predict_cla].item()
#
#         # 在图片上显示预测结果
#         plt.text(200, -8, f'Pred: {predicted_class}\nAcc: {accuracy:.4f}', fontsize=12,
#                  color='red', bbox=dict(facecolor='yellow', alpha=0.5))
#
#         # 写入真实类别、预测类别和准确率到文件
#         file.write(f'Trueclass: {true_label}  Predictclass: {predicted_class}  Accuracy: {accuracy:.4f}\n')
#
#         # 打印到控制台
#         print(f'{i} Trueclass: {true_label}, Predictclass: {predicted_class}, Accuracy: {accuracy:.4f}')
#
#         # 显示图片
#         plt.show()





# # 可以循环遍历一遍的
# import torch
# import random
# from model import ZFNet
# from torchvision import transforms
# import json
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import glob
#
# def load_images_from_class_folder(folder):
#     """Loads images and filenames from a single class folder."""
#     images = []
#     filenames = []
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#         try:
#             img = Image.open(img_path).convert('RGB')  # Ensure RGB format
#             img = img.resize((224, 224))
#             img = np.array(img)
#             images.append(img)
#             filenames.append(filename)
#         except (IOError, OSError) as e:
#             print(f"Error loading image {img_path}: {e}")
#     return images, filenames
#
#
# # Specify the root folder path (containing subfolders for each class)
# root_folder_path = r"C:\Users\35008\Desktop\val"  # Replace with your path
#
# # Get paths to all image files within each class subfolder
# class_image_data = {}
# for class_name in os.listdir(root_folder_path):
#     class_path = os.path.join(root_folder_path, class_name)
#     if os.path.isdir(class_path):
#         images, filenames = load_images_from_class_folder(class_path)
#         class_image_data[class_name] = {"images": images, "filenames": filenames}
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# data_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]
# )
#
# # Load class indices mapping
# try:
#     with open('./class_indices.json', 'r') as f:
#         class_indict = json.load(f)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# # Create model and load pretrained weights
# model = ZFNet(num_classes=len(class_indict))  # Adjust num_classes if needed
# model_weight_path = "./AlexNet.pth"
# try:
#     model.load_state_dict(torch.load(model_weight_path))
# except Exception as e:
#     print(f"Error loading model weights: {e}")
#     exit(-1)
# model.eval()
#
#
# with open('input.txt', 'a', encoding='utf-8') as file:
#     for class_name, data in class_image_data.items():
#         images = data['images']
#         filenames = data['filenames']
#         for i, (img, img_filename) in enumerate(zip(images, filenames)):
#             try:
#                 true_label = class_name  # Class name is the true label
#
#                 img = cv2.resize(img, (224, 224))
#                 img_tensor = data_transform(img)
#                 img_tensor = torch.unsqueeze(img_tensor, dim=0)
#
#                 with torch.no_grad():
#                     output = torch.squeeze(model(img_tensor))
#                     predict = torch.softmax(output, dim=0)
#                     predict_cla = torch.argmax(predict).numpy()
#                     predicted_class = class_indict[str(predict_cla)]
#                     accuracy = predict[predict_cla].item()
#
#                 file.write(f'Trueclass: {true_label}  Predictclass: {predicted_class}  Accuracy: {accuracy:.4f}\n')
#                 print(f'Trueclass: {true_label}, Predictclass: {predicted_class}, Accuracy: {accuracy:.4f}')
#
#             except Exception as e:
#                 print(f"Error processing image from class '{class_name}': {e}")
#
# print("Prediction complete!")


# import torch
# import random
# from model import ZFNet
# from torchvision import transforms
# import json
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import glob
#
# def load_images_from_folder(folder, max_images_per_class=float('inf')):
#     """Loads images and filenames from a single class folder."""
#     images = []
#     filenames = []
#     count = 0
#     for filename in os.listdir(folder):
#         if count >= max_images_per_class:
#             break
#         img_path = os.path.join(folder, filename)
#         try:
#             img = Image.open(img_path).convert('RGB')
#             img = img.resize((224, 224))
#             img = np.array(img)
#             images.append(img)
#             filenames.append(filename)
#             count += 1
#         except (IOError, OSError) as e:
#             print(f"Error loading image {img_path}: {e}")
#     return images, filenames
#
# # Specify the root folder path (containing subfolders for each class)
# root_folder_path = r"C:\Users\35008\Desktop\cardataset - 副本"  # Replace with your path
#
# # Get paths to all image files within each class subfolder
# class_image_data = {}
# max_images_per_class = 300  # Fixed number of images per class (adjust as needed)
#
# for class_name in os.listdir(root_folder_path):
#     class_path = os.path.join(root_folder_path, class_name)
#     if os.path.isdir(class_path):
#         images, filenames = load_images_from_folder(class_path, max_images_per_class=max_images_per_class)
#         class_image_data[class_name] = {"images": images, "filenames": filenames}
#
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# data_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ]
# )
#
# # Load class indices mapping (crucial)
# try:
#     with open('./class_indices.json', 'r') as f:
#         class_indict = json.load(f)
# except FileNotFoundError:
#     print("Error: class_indices.json not found.")
#     exit(1)  # Exit with a non-zero code to indicate error
#
#
# model = ZFNet(num_classes=len(class_indict))  # Crucial: Use the correct number of classes
# model_weight_path = "./AlexNet.pth"  # Or your model's weights path
# try:
#     model.load_state_dict(torch.load(model_weight_path))
# except FileNotFoundError:
#     print(f"Error: Model weights file '{model_weight_path}' not found.")
#     exit(1)
# model.eval()
#
#
# with open('predictions.txt', 'a', encoding='utf-8') as file:
#     for class_name, data in class_image_data.items():
#         images = data['images']
#         filenames = data['filenames']
#
#         for i, img in enumerate(images):
#             try:
#                 img_tensor = data_transform(img)
#                 img_tensor = torch.unsqueeze(img_tensor, dim=0)
#                 with torch.no_grad():
#                     output = model(img_tensor)
#                     predict = torch.softmax(output, dim=1)
#                     predict_cla = torch.argmax(predict, dim=1).item()
#                     predicted_class = list(class_indict.values())[predict_cla]
#                     accuracy = predict[0][predict_cla].item()
#
#                 file.write(f'Trueclass: {class_name}  Predictclass: {predicted_class}  Accuracy: {accuracy:.4f}\n')
#                 print(f'Trueclass: {class_name}, Predictclass: {predicted_class}, Accuracy: {accuracy:.4f}')
#
#             except Exception as e:
#                 print(f"Error processing image from class '{class_name}': {e}")
#
# print("Prediction complete!")

# 比较好的好多个1的
import torch
import random
from model import ZFNet
from torchvision import transforms
import json
import cv2
import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder):
    images = []
    filenames = []  # 用于存储图片的文件名
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = Image.open(img_path)
        img = img.resize((224, 224))  # 你可以根据需要调整图片大小
        img = np.array(img)
        images.append(img)
        filenames.append(filename)  # 保存图片文件名

    images = np.array(images)
    return images, filenames


# 指定包含图片的文件夹路径
folder_path = r"D:\智能信息处理学习\firstessay\数据集\cars - 副本"

# 加载图片数据集和文件名
image_dataset, filenames = load_images_from_folder(folder_path)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]
)

# 加载类索引映射
try:
    json_file = open('./class_indices.json', 'r')
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# 创建模型并加载预训练权重
model = ZFNet(num_classes=4)
model_weight_path = "./AlexNet.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

with open('input.txt', 'a', encoding='utf-8') as file:
    for i in range(1000):
        # 随机选择一张图片
        ss = random.randint(0, len(image_dataset) - 1)
        img = image_dataset[ss]
        img_filename = filenames[ss]  # 获取图片的文件名

        # 假设文件名就是类别信息（例如：'cat_001.jpg'）
        true_label = img_filename.split('_')[0]  # 获取文件名的前缀作为真实标签

        img = cv2.resize(img, (224, 224))

        # 对图像进行预处理
        img_tensor = data_transform(img)
        img_tensor = torch.unsqueeze(img_tensor, dim=0)

        # 进行预测
        with torch.no_grad():
            output = torch.squeeze(model(img_tensor))
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()

        # 获取预测的类别名称和准确率
        predicted_class = class_indict[str(predict_cla)]  # 使用类别名称而不是索引
        accuracy = predict[predict_cla].item()

        # 写入真实类别、预测类别和准确率到文件
        file.write(f'Trueclass: {true_label}  Predictclass: {predicted_class}  Accuracy: {accuracy:.4f}\n')

        # 打印到控制台
        print(f'{i} Trueclass: {true_label}, Predictclass: {predicted_class}, Accuracy: {accuracy:.4f}')



# # 调过之后差一点的
# import torch
# import random
# from model import ZFNet
# from torchvision import transforms
# import json
# import cv2
# import os
# import numpy as np
# from PIL import Image
#
# def load_images_from_folder(folder):
#     images = []
#     filenames = []  # 用于存储图片的文件名
#     for filename in os.listdir(folder):
#         img_path = os.path.join(folder, filename)
#         img = Image.open(img_path)
#         img = img.resize((224, 224))  # 你可以根据需要调整图片大小
#         img = np.array(img)
#         images.append(img)
#         filenames.append(filename)  # 保存图片文件名
#
#     images = np.array(images)
#     return images, filenames
#
# # 指定包含图片的文件夹路径
# folder_path = r"C:\Users\35008\Desktop\cars - 副本"
#
# # 加载图片数据集和文件名
# image_dataset, filenames = load_images_from_folder(folder_path)
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# data_transform = transforms.Compose(
#     [
#         transforms.ToTensor(),
#     ]
# )
#
# # 加载类索引映射
# try:
#     json_file = open('./class_indices.json', 'r')
#     class_indict = json.load(json_file)
# except Exception as e:
#     print(e)
#     exit(-1)
#
# # 创建模型并加载预训练权重
# model = ZFNet(num_classes=4)
# model_weight_path = "./AlexNet.pth"
# model.load_state_dict(torch.load(model_weight_path))
# model.eval()
#
# with open('input.txt', 'a', encoding='utf-8') as file:
#     for i in range(1000):
#         # 随机选择一张图片
#         ss = random.randint(0, len(image_dataset) - 1)
#         img = image_dataset[ss]
#         img_filename = filenames[ss]  # 获取图片的文件名
#
#         # 在图像上加入噪声
#         noise = np.random.normal(0, 0.1, img.shape).astype(np.float32)
#         img = cv2.add(img, noise, dtype=cv2.CV_32F)
#         img = np.clip(img, 0, 255).astype(np.uint8)  # 确保像素值在有效范围内
#
#         # 随机翻转或旋转图像
#         if random.random() < 0.5:
#             img = cv2.flip(img, 1)  # 水平翻转
#         if random.random() < 0.5:
#             angle = random.randint(-30, 30)  # 随机旋转角度
#             h, w = img.shape[:2]
#             M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
#             img = cv2.warpAffine(img, M, (w, h))
#
#         img = cv2.resize(img, (224, 224))
#
#         # 对图像进行预处理
#         img_tensor = data_transform(img)
#         img_tensor = torch.unsqueeze(img_tensor, dim=0)
#
#         # 进行预测
#         with torch.no_grad():
#             output = torch.squeeze(model(img_tensor))
#             predict = torch.softmax(output, dim=0)
#             predict_cla = torch.argmax(predict).numpy()
#
#         # 获取预测的类别名称和准确率
#         predicted_class = class_indict[str(predict_cla)]  # 使用类别名称而不是索引
#         accuracy = predict[predict_cla].item()
#
#         # 写入真实类别、预测类别和准确率到文件
#         file.write(f'Trueclass: {img_filename.split("_")[0]}  Predictclass: {predicted_class}  Accuracy: {accuracy:.4f}\n')
#
#         # 打印到控制台
#         print(f'{i} Trueclass: {img_filename.split("_")[0]}, Predictclass: {predicted_class}, Accuracy: {accuracy:.4f}')
#
#












# import torch
# import random
# from model import ZFNet
# from torchvision import transforms
# import json
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt
#
# def load_images_from_folder(folder):
#     images = []
#     # 假设文件夹中的文件名包含了标签信息（例如：image_1_cat.jpg 表示类别为 cat）
#     # 这里为了简化，我们不加载标签，只加载图像
#     for filename in os.listdir(folder):
#         img = Image.open(os.path.join(folder, filename))
#         img = img.resize((224, 224))
#         images.append(img)
#     return images
#
# folder_path = r"D:\智能信息处理学习\firstessay\车牌数据集\cardatabase\Bluecar"
# image_dataset = load_images_from_folder(folder_path)
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#
# data_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])
#
# # 打开 input.txt 文件以写入
# with open('input.txt', 'a') as file:
#     for i in range(5):
#         ss = random.randint(0, len(image_dataset) - 1)  # 确保索引在有效范围内（从0开始）
#         img = image_dataset[ss]
#         img_tensor = data_transform(img).unsqueeze(0)  # 直接转换为 tensor 并增加 batch 维度
#
#         try:
#             with open('./class_indices.json', 'r') as json_file:
#                 class_indict = json.load(json_file)
#         except Exception as e:
#             print(e)
#             continue  # 跳过当前迭代并继续下一次
#
#         model = ZFNet(num_classes=len(class_indict))
#         model_weight_path = "./AlexNet.pth"  # 确保文件名和路径正确，且与 ZFNet 模型匹配
#         model.load_state_dict(torch.load(model_weight_path))
#         model.eval()
#         with torch.no_grad():
#             output = model(img_tensor)
#             predict = torch.softmax(output, dim=1)
#             predict_cla = torch.argmax(predict).item()
#             predict_acc = predict[predict_cla].item()
#
#             # 格式化输出并写入文件
#             result_line = f"预测：{class_indict[str(predict_cla)]}\t预测准确率：{predict_acc:.4f}\n"
#             file.write(result_line)
#
#             # 显示图片和预测结果
#             img_np = img_tensor.permute(1, 2, 0).numpy() * 255  # 转换回图像格式
#             img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # 确保值在有效范围内
#             plt.imshow(img_np)
#             plt.text(20, 20, f'Predict: {class_indict[str(predict_cla)]}\nAccuracy: {predict_acc:.4f}', fontsize=12,
#                      color='red', bbox=dict(facecolor='yellow', alpha=0.5))
#             plt.show()