import torch.nn as nn
from torchvision import models

class DRModel(nn.Module):
    def __init__(self, num_classes=5, pretrain=False):
        super(DRModel, self).__init__()
        self.backbone = models.resnet50(weights='IMAGENET1K_V1' if pretrain else None)
        dim_mlp = self.backbone.fc.in_features
        
        # SimCLR Projection Head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, 128)
        )
        
        # Final Classification Head
        self.classifier = nn.Linear(dim_mlp, num_classes)
        self.backbone.fc = nn.Identity() # Backbone output is now 2048D vector

    def forward(self, x, mode='classify'):
        features = self.backbone(x)
        if mode == 'contrastive':
            return self.projector(features)
        return self.classifier(features)