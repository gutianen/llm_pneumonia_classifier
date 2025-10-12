
"""
肺炎图片：图片分类训练， 包含数据加载、模型训练、验证、早停、断点续训、可视化结果、预测样本
"""

import os  # 操作系统接口
import random  # 随机数生成
import numpy as np  # 数值计算
import matplotlib.pyplot as plt  # 数据可视化
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器
from torchvision import transforms, models  # 计算机视觉相关
import re
from BaseImageClassifier import BaseImageDataset, BaseImageDataLoader, BaseImageClassifier, default_config


# ==== 画图设置 ====
plt.rcParams['font.size'] = 16 # 设置全局字体大小为16
plt.rcParams['font.sans-serif'] = ['STHeiti']
plt.rcParams['axes.unicode_minus'] = False# 解决负号显示为方块的问题

# ==== 初始化设置 ====
SEED = 42
torch.manual_seed(SEED)  # 设置PyTorch随机种子
np.random.seed(SEED)  # 设置NumPy随机种子
random.seed(SEED)  # 设置Python随机种子
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.95)   # 限制最多使用95%内存
    torch.backends.cuda.matmul.allow_tf32 = True       # 开启 TF32 矩阵乘法加速
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True  # 启用cudnn自动优化
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动检测设备
print(f"Using device: {device}")  # 打印使用的设备


# ==== 参数设置 ====
config = {
    'training_name': 'pneumonia_model_traning',
    'data_dir': r'data/COVID-19_Radiography_Dataset',
    'model_name': 'resnet18',
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.0005,
    'lr_reduce_patience': 2,
    'lr_reduce_factor': 0.85,
    'weight_decay': 1e-4,
    'drop_out': 0.3,
    'num_of_workers': 8,
    'label_smoothing': 0.1,
    'early_stop_patience': 20,
    'model_file_name': 'pneumonia_model.pth'
}


# ==== 自定义类 ====
class PneumoniaImageDataset(BaseImageDataset):
    """处理自定义图像数据集，支持从文件夹自动分类，支持文件名格式：类别-编号.jpg"""
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集根目录
        :param transform: 图像变换函数
        """
        self.root_dir = root_dir
        self.transform = transform

        print(os.listdir(root_dir))
        valid_images = []
        # 递归遍历所有子目录
        for dirpath, _, filenames in os.walk(root_dir):
            # 筛选当前目录下的图片文件
            for f in filenames:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # 拼接完整路径（可选，根据需求决定是否保留）
                    full_path = os.path.join(dirpath, f)
                    valid_images.append(full_path)  # 存储完整路径
                    # 如果只需要文件名，可用：valid_images.append(f)
        print('valid_images.size: ', len(valid_images))

        # 从文件名提取类别（假设格式为"类别-编号.扩展名"）
        self.classes = sorted(list(set([os.path.basename(f).split('-')[0] for f in valid_images])))

        # 创建类别到索引的映射
        self.classes_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print('classes', self.classes)
        print('classes_to_idx', self.classes_to_idx)

        # 构建图像路径和标签列表
        self.samples = []
        for img_full_path in valid_images:
            # 提取类别
            cls = os.path.basename(img_full_path).split('-')[0]   # COVID-970.png -》 COVID
            self.samples.append((img_full_path, self.classes_to_idx[cls]))

        print('第一条image', self.samples[0][0], self.samples[0][1])

class PneumoniaImageDataLoader(BaseImageDataLoader):
    def __init__(self, data_dir, custom_image_dataset=None):
        super().__init__(data_dir, custom_image_dataset)

    def image_transform(self):
        # 数据增强变换
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(300, scale=(0.7, 1.0)),  # 随机裁剪缩放(输出300x300)
            transforms.RandomHorizontalFlip(),  # 50%概率水平翻转
            transforms.RandomVerticalFlip(),  # 50%概率垂直翻转
            transforms.RandomRotation(45),  # 随机旋转±45度
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),  # 颜色抖动(亮度、对比度、饱和度各30%，色相10%)
            transforms.ToTensor(),  # 转换为张量
            # ImageNet标准归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(330),  # 调整大小保持比例
            transforms.CenterCrop(300),  # 中心裁剪300x300
            transforms.ToTensor(),  # 转换为张量
            # 相同归一化
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return train_transform, test_transform


# ==== 模型初始化 ====
def init_model(num_of_classes=10):
    model = models.resnet18(pretrained=True)  # 加载预训练的ResNet18
    num_features = model.fc.in_features  # 获取原分类器的输入特征数
    # 替换分类器最后一层
    model.fc = nn.Sequential(
        nn.Dropout(config['drop_out']),  # 30%的dropout
        nn.Linear(num_features, num_of_classes)  # 全连接层
    )

    return model

# ==== 主流程 ====
def main():
    # 加载训练和测试数据
    pneumoniaDataset = PneumoniaImageDataset(config['data_dir'])
    pneumoniaDataLoader = PneumoniaImageDataLoader(config['data_dir'], custom_image_dataset=pneumoniaDataset)
    train_loader, test_loader, classes = pneumoniaDataLoader.load_data(config['batch_size'], config['num_of_workers'])

    # 初始化模型
    model = init_model(len(classes))

    # 交叉熵损失
    criterion = nn.CrossEntropyLoss(label_smoothing=config['label_smoothing'])

    # AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'])

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=config['lr_reduce_patience'],
                                                     factor=config['lr_reduce_factor'])
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer,
    #     T_max=config['epochs'],  # 周期为总轮数
    #     eta_min=config['learning_rate'] * 0.01  # 最小学习率
    # )

    # 初始化分类器对象
    classifier = BaseImageClassifier(device, training_name=config['training_name'], num_of_classes=len(classes), model=model, criterion=criterion, optimizer=optimizer, scheduler=scheduler, model_file_name=config['model_file_name'], early_stop_patience=config['early_stop_patience'])

    # 训练模型
    model, history = classifier.train_model(train_loader, test_loader, config['epochs'])

    # 最终评估
    final_loss, final_acc, y_test, y_pred = classifier.evaluate(test_loader)
    print(f"最终测试准确率: {final_acc:.2f}%")
    print(f"最终测试损失: {final_loss:.6f}")
    classifier.print_result_report(y_test, y_pred)
    classifier.visualize_confusion_matrix(y_test, y_pred)

    # 可视化结果
    classifier.visualize_results(history)

    # 加载最佳模型进行预测可视化
    classifier.checkpoint = torch.load(config['model_file_name'])
    classifier.model.load_state_dict(classifier.checkpoint['model_state'])
    classifier.visualize_predictions(test_loader, classes, 12)

if __name__ == '__main__':
    main()
