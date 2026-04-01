import torch
import torch.nn as nn
from torchvision import models

class StageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = models.resnet50(weights="IMAGENET1K_V2")
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        return self.classifier(x)
