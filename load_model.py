# Создайте файл load_model.py и вставьте в него ваш код
import torch
import torch.nn as nn
from torchvision.models import resnet18
import os

# Определение класса модели
class EnhancedModel(nn.Module):
    def __init__(self, n_classes: int):
        super(EnhancedModel, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        return self.resnet(x)

# Создание экземпляра модели
model = EnhancedModel(n_classes=10)  # Количество классов должно совпадать с Colab

# Загрузка весов
model_path = 'enhanced_model.pth'
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print("Модель успешно загружена.")
else:
    print(f"Ошибка: файл '{model_path}' не найден.")

