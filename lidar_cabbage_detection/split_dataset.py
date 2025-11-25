import os
import random
import shutil

# 设置路径
image_dir = os.path.expanduser('~/dataset/images')  # 图像目录
label_dir = os.path.expanduser('~/dataset/labels')  # 标签目录

# 设置划分比例
train_ratio = 0.8
val_ratio = 0.2

# 获取所有图片文件（假设图片为 .jpg, .png 格式）
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 随机打乱文件顺序
random.shuffle(image_files)

# 计算训练集和验证集的数量
train_size = int(len(image_files) * train_ratio)
val_size = len(image_files) - train_size

# 划分训练集和验证集
train_files = image_files[:train_size]
val_files = image_files[train_size:]

# 创建训练集和验证集文件夹
train_image_dir = os.path.expanduser('~/dataset/images/train')
val_image_dir = os.path.expanduser('~/dataset/images/val')
train_label_dir = os.path.expanduser('~/dataset/labels/train')
val_label_dir = os.path.expanduser('~/dataset/labels/val')

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# 将文件移动到训练集和验证集文件夹
for image_file in train_files:
    # 复制图像和标签
    shutil.copy(os.path.join(image_dir, image_file), os.path.join(train_image_dir, image_file))
    label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(train_label_dir, label_file))

for image_file in val_files:
    # 复制图像和标签
    shutil.copy(os.path.join(image_dir, image_file), os.path.join(val_image_dir, image_file))
    label_file = image_file.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt')
    shutil.copy(os.path.join(label_dir, label_file), os.path.join(val_label_dir, label_file))

print(f"数据集划分完成：训练集 {len(train_files)} 张图片，验证集 {len(val_files)} 张图片")

