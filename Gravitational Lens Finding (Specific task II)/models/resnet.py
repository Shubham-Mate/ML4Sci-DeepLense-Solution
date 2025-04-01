import torch
from torchvision import models

class ResNetFineTune(torch.nn.Module):
    def __init__(self, num_classes=3, freeze_blocks=2):
        super(ResNetFineTune, self).__init__()
        
        # Load Pretrained ResNet
        self.base_model = models.resnet18(pretrained=True) 
        
        # Freeze first few blocks to retain general feature extraction
        num_layers_to_freeze = freeze_blocks * 2  # Each ResNet block contains 2 layers
        layers = list(self.base_model.children())[:num_layers_to_freeze]
        for layer in layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.base_model = torch.nn.Sequential(*list(self.base_model.children())[:-2])  # Keep up to last conv layer
        
        self.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        # Custom Classification Head
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(512, 256),  
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.base_model(x)  # Feature extraction
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)  # Custom head
        return x