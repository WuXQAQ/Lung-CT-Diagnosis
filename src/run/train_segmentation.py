import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils.dataset import SegmentationDataset
from src.models.unet import UNet

# 损失函数
def dice_loss(pred, target):
    smooth = 1.0
    intersection = (pred * target).sum()
    return 1 - ((2. * intersection + smooth) / (pred.sum() + target.sum() + smooth))

# 数据加载
train_dataset = SegmentationDataset(image_dir="./data/train/images", mask_dir="./data/train/masks")
val_dataset = SegmentationDataset(image_dir="./data/val/images", mask_dir="./data/val/masks")
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# 初始化模型和优化器
model = UNet()
optimizer = Adam(model.parameters(), lr=1e-3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

best_val_loss = float('inf')  # 保存最好的验证损失

# 训练循环
EPOCHS = 20
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    # 进度条
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as t:
        for images, masks in t:
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            _, outputs = model(images)
            loss = dice_loss(outputs, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            t.set_postfix(loss=train_loss / (t.n + 1))

    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss/len(train_loader):.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            # 前向传播
            _, outputs = model(images)
            loss = dice_loss(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # 保存验证集损失最低的模型
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "./models/unet_best.pth")
        print("Best model saved.")

# 保存最终模型
torch.save(model.state_dict(), "./models/unet.pth")

