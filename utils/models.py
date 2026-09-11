import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, ResNet

def get_model(dataset_name, model):
    if model == "CNN":
        if dataset_name in ["Fashion-MNIST"]:
            return FashionMNISTCNN()
        elif dataset_name in ["CIFAR10", "CINIC-10"]:
            return CifarCNN()
    elif model == "ResNet18":
        if dataset_name in ["CIFAR10", "CINIC-10"]:
            return CifarResNet(num_classes=10)
    raise ValueError(f"Unknown model or dataset: {model}, {dataset_name}")

class FashionMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 5 * 5, 128),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(128, 10)

    def forward(self, x, return_feat=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)
        if return_feat:
            return out, feature
        return out

class CifarResNet(ResNet):
    def __init__(self, num_classes=10):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.maxpool = nn.Identity()

    def forward(self, x, return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        out = self.fc(features)
        if return_feat:
            return out, features
        return out

class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(3, 16, 5, padding=0),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 5, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 128),
            nn.LeakyReLU()
        )
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x, return_feat=False):
        feature = self.hidden_layers(x)
        out = self.fc(feature)
        if return_feat:
            return out, feature
        return out