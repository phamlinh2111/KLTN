import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms

class FeatureExtractor(nn.Module):
    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

class EfficientNetGen(FeatureExtractor):
    def __init__(self, model: str):
        super(EfficientNetGen, self).__init__()

        self.efficientnet = EfficientNet.from_pretrained(model)
        self.dropout = nn.Dropout(p=0.3) 
        self.classifier = nn.Linear(self.efficientnet._conv_head.out_channels, 1)
        del self.efficientnet._fc

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.dropout(x) 
        x = self.classifier(x)
        return x
        
class EfficientNetB0(EfficientNetGen):
    def __init__(self):
        super(EfficientNetB0, self).__init__(model='efficientnet-b0')        
