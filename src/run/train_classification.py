import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from src.utils.dataset import get_data_loaders
from src.models.unet import UNet
from src.models.resnet50_classifier import ResNet50Classifier
from tqdm import tqdm

# 加载训练好的 U-Net 模型
unet = UNet()
unet.load_state_dict(torch.load("./models/unet.pth"))
unet.eval()  # 设置为评估模式
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = unet.to(device)

# 初始化 ResNet-50 分类模型
model = ResNet50Classifier(input_channels=128, num_classes=2)
model = model.to(device)

# 数据加载
train_loader, val_loader = get_data_loaders(data_dir="./data", batch_size=32)

# 优化器和损失函数
optimizer = Adam(model.parameters(), lr=1e-4)
criterion = CrossEntropyLoss()

# 训练循环
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    # 进度条显示
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as t:
        for images, labels in t:
            images, labels = images.to(device), labels.to(device)

            # 提取 Encoder 特征
            with torch.no_grad():  # Encoder 不参与更新
                features, _ = unet(images)

            # 前向传播到 ResNet
            outputs = model(features)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 更新进度条
            t.set_postfix(loss=train_loss / (t.n + 1), accuracy=100. * correct / total)

    # 打印训练结果
    train_accuracy = 100. * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}, Train Accuracy: {train_accuracy:.2f}%")

    # 验证阶段
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            # 提取 Encoder 特征
            features, _ = unet(images)

            # 前向传播到 ResNet
            outputs = model(features)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_accuracy = 100. * correct / total
    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# 保存分类模型
torch.save(model.state_dict(), "./models/resnet50_classifier.pth")
