# import os
# from shutil import copy
# import random
#
#
# def mkfile(file):
#     if not os.path.exists(file):
#         os.makedirs(file)
#
#
# file = r"C:\Users\35008\Desktop\cardataset"
# flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
# mkfile('car_data/train')
# for cla in flower_class:
#     mkfile('car_data/train/'+cla)
#
# mkfile('car_data/val')
# for cla in flower_class:
#     mkfile('car_data/val/'+cla)
#
# split_rate = 0.1
# for cla in flower_class:
#     cla_path = file + '/' + cla + '/'
#     images = os.listdir(cla_path)
#     num = len(images)
#     eval_index = random.sample(images, k=int(num*split_rate))
#     for index, image in enumerate(images):
#         if image in eval_index:
#             image_path = cla_path + image
#             new_path = 'car_data/val/' + cla
#             copy(image_path, new_path)
#         else:
#             image_path = cla_path + image
#             new_path = 'car_data/train/' + cla
#             copy(image_path, new_path)
#         print("\r[{}] processing [{}/{}]".format(cla, index+1, num), end="")  # processing bar
#     print()
#
# print("processing done!")
#
#
#
# import os
# from shutil import copy
# import random
# 
# 
# def mkfile(file):
#     if not os.path.exists(file):
#         os.makedirs(file)
# 
# 
# # 数据集文件夹路径
# file = r"C:\Users\35008\Desktop\cardataset"
# flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
# 
# # 创建训练集和验证集的目录结构
# mkfile('car_data/train')
# for cla in flower_class:
#     mkfile(f'car_data/train/{cla}')
# 
# mkfile('car_data/val')
# for cla in flower_class:
#     mkfile(f'car_data/val/{cla}')
# 
# # 分割比例 3:1
# split_rate = 0.25  # 验证集占总数据的 25%
# 
# # 遍历每个类别
# for cla in flower_class:
#     cla_path = os.path.join(file, cla)
#     images = os.listdir(cla_path)
#     num = len(images)
# 
#     # 随机打乱图片顺序
#     random.shuffle(images)
# 
#     # 计算分割点
#     val_size = int(num * split_rate)
# 
#     # 将前 val_size 张图片分配到验证集，剩余分配到训练集
#     val_images = images[:val_size]
#     train_images = images[val_size:]
# 
#     # 将图片复制到训练集和验证集
#     for image in val_images:
#         image_path = os.path.join(cla_path, image)
#         new_path = os.path.join('car_data/val', cla, image)
#         copy(image_path, new_path)
# 
#     for image in train_images:
#         image_path = os.path.join(cla_path, image)
#         new_path = os.path.join('car_data/train', cla, image)
#         copy(image_path, new_path)
# 
#     # 输出进度
#     print(f"[{cla}] - Processing {num} images: {len(train_images)} for train, {len(val_images)} for val.")
# 
# print("Processing done!")
# 
# import os
# from shutil import copy
# import random
#
#
# def mkfile(file):
#     if not os.path.exists(file):
#         os.makedirs(file)
#
#
# # Path to the original dataset
# file = r"C:\Users\35008\Desktop\cardataset"
#
# # Get the list of categories (folders) in the dataset, excluding any .txt files
# flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]
#
# # Create the car_data folder and its subfolders for train, val, and test
# mkfile('cardata')
# mkfile('cardata/train')
# mkfile('cardata/val')
# mkfile('cardata/test')
#
# # Create category subfolders inside train, val, and test
# for cla in flower_class:
#     mkfile(f'cardata/train/{cla}')
#     mkfile(f'cardata/val/{cla}')
#     mkfile(f'cardata/test/{cla}')
#
# # Split ratio: 3:1:1 (train:val:test)
# split_rate_train = 0.6
# split_rate_val = 0.2
# split_rate_test = 0.2
#
# # Split the images into train, val, and test sets for each class
# for cla in flower_class:
#     cla_path = os.path.join(file, cla)  # Path to the class folder
#     images = os.listdir(cla_path)  # List all images in the class folder
#     num = len(images)
#
#     # Randomly shuffle the images
#     random.shuffle(images)
#
#     # Split the images based on the defined ratios
#     num_train = int(num * split_rate_train)
#     num_val = int(num * split_rate_val)
#
#     # Create the splits
#     train_images = images[:num_train]
#     val_images = images[num_train:num_train + num_val]
#     test_images = images[num_train + num_val:]
#
#     # Copy the images into the corresponding folders
#     for index, image in enumerate(train_images):
#         image_path = os.path.join(cla_path, image)
#         new_path = os.path.join('cardata/train', cla, image)
#         copy(image_path, new_path)
#         print(f"\r[train] {cla} processing [{index + 1}/{len(train_images)}]", end="")
#
#     for index, image in enumerate(val_images):
#         image_path = os.path.join(cla_path, image)
#         new_path = os.path.join('cardata/val', cla, image)
#         copy(image_path, new_path)
#         print(f"\r[val] {cla} processing [{index + 1}/{len(val_images)}]", end="")
#
#     for index, image in enumerate(test_images):
#         image_path = os.path.join(cla_path, image)
#         new_path = os.path.join('cardata/test', cla, image)
#         copy(image_path, new_path)
#         print(f"\r[test] {cla} processing [{index + 1}/{len(test_images)}]", end="")
#
#     print()  # New line after each class processing
#
# print("Processing done!")
import os
from shutil import copy
import random

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

# Path to the original dataset
file = r"D:\智能信息处理学习\firstessay\车牌数据集\cardatabase"

# Get the list of categories (folders) in the dataset, excluding any .txt files
flower_class = [cla for cla in os.listdir(file) if ".txt" not in cla]

# Create the cardata folder and its subfolders for train, val, and test
mkfile('cardata')
mkfile('cardata/train')
mkfile('cardata/val')
mkfile('cardata/test')

# Create category subfolders inside train, val, and test
for cla in flower_class:
    mkfile(f'cardata/train/{cla}')
    mkfile(f'cardata/val/{cla}')
    mkfile(f'cardata/test/{cla}')

# Split ratio: 3:1:1 (train:val:test)
split_rate_train = 0.6
split_rate_val = 0.2
split_rate_test = 0.2

# Split the images into train, val, and test sets for each class
for cla in flower_class:
    cla_path = os.path.join(file, cla)  # Path to the class folder
    images = os.listdir(cla_path)       # List all images in the class folder
    num_images = len(images)

    # Randomly shuffle the images
    random.shuffle(images)

    # Calculate the split size based on the 3:1:1 ratio
    num_train = int(num_images * split_rate_train)
    num_val = int(num_images * split_rate_val)
    num_test = num_images - num_train - num_val  # Ensure that the total sum matches

    # Adjust the split if needed (due to rounding issues)
    if num_train + num_val + num_test != num_images:
        difference = num_images - (num_train + num_val + num_test)
        num_test += difference  # Add the leftover images to the test set

    # Create the splits
    train_images = images[:num_train]
    val_images = images[num_train:num_train + num_val]
    test_images = images[num_train + num_val:]

    # Copy the images into the corresponding folders
    for index, image in enumerate(train_images):
        image_path = os.path.join(cla_path, image)
        new_path = os.path.join('cardata/train', cla, image)
        copy(image_path, new_path)
        print(f"\r[train] {cla} processing [{index+1}/{len(train_images)}]", end="")

    for index, image in enumerate(val_images):
        image_path = os.path.join(cla_path, image)
        new_path = os.path.join('cardata/val', cla, image)
        copy(image_path, new_path)
        print(f"\r[val] {cla} processing [{index+1}/{len(val_images)}]", end="")

    for index, image in enumerate(test_images):
        image_path = os.path.join(cla_path, image)
        new_path = os.path.join('cardata/test', cla, image)
        copy(image_path, new_path)
        print(f"\r[test] {cla} processing [{index+1}/{len(test_images)}]", end="")

    print()  # New line after each class processing

print("Processing done!")
