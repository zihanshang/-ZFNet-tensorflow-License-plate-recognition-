import os  #导入模块

filename = r'C:\Users\35008\Desktop\cardataset - 副本\bluecar' #文件地址
list_path = os.listdir(filename)   #读取文件夹里面的名字

count = 1
for index in list_path:
    path = filename + '\\' + index  # 原本文件名
    new_path = filename + '\\' + f'carblue {count}.jpg'
    print(new_path)
    os.rename(path, new_path)
    count += 1

print('修改完成')

# import os
# # 正确的文件夹路径
# folder_path = r'C:\Users\35008\Desktop\cardataset\bluecar'
# # 获取文件夹中所有文件的列表，不包括子文件夹
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# # 开始重命名
# count = 1
# for file in files:
#     # 构建完整的旧文件路径
#     old_path = os.path.join(folder_path, file)
#     # 构建新的文件路径，使用格式化字符串确保文件扩展名保留
#     file_base, file_ext = os.path.splitext(file)
#     new_path = os.path.join(folder_path, f'{count}bluecar{file_ext}')
#
#     # # 检查新文件名是否已存在，如果存在则跳过或处理冲突
#     # while os.path.exists(new_path):
#     #     count += 1
#     #     new_path = os.path.join(folder_path, f'{count}bluecar{file_ext}')
#     #
#     #     # 重命名文件
#     # os.rename(old_path, new_path)
#     count += 1
# print('修改完成')


# import os
# # 正确的文件夹路径
# folder_path = r'C:\Users\35008\Desktop\cardataset\bluecar'
# # 获取文件夹中所有文件的列表，不包括子文件夹
# files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
# # 开始重命名
# count = 1
# for file in files:
#     # 构建完整的旧文件路径
#     old_path = os.path.join(folder_path, file)
#     # 获取文件的扩展名
#     file_ext = os.path.splitext(file)[1]
#     # 尝试重命名文件，直到找到一个不冲突的新文件名
#     new_file_base = f'{count}bluecar'
#     while True:
#         new_path = os.path.join(folder_path, new_file_base + file_ext)
#         if not os.path.exists(new_path):
#             # 如果新文件名不存在，则进行重命名并跳出循环
#             os.rename(old_path, new_path)
#             break
#             # 如果新文件名已存在，则递增count并尝试下一个文件名
#         count += 1
#         new_file_base = f'{count}bluecar'
#         # 注意：这里的count不需要在每次迭代结束时都增加，因为它是在while循环内部根据需要增加的。
# # 循环结束后打印完成信息
# print('修改完成')