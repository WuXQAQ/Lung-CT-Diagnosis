from torchvision.models import resnet50
import torch.nn as nn

class ResNet50Classifier(nn.Module):
    def __init__(self, input_channels=128, num_classes=2):
        super(ResNet50Classifier, self).__init__()
        self.resnet = resnet50(pretrained=True)
        # 修改第一个卷积层以适应 Encoder 输出
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的全连接层以适应目标类别数
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)
