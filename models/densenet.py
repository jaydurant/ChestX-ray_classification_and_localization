import torch.nn as nn
import torchvision.models as models

class DenseNet(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet, self).__init__()
        densenet = models.densenet121(pretrained=True)
        densenet.fc = nn.Linear(in_features=densenet.fc.in_features, out_features=n_classes)
        self.base_model = densenet
    
    def forward(self, x):
        x = self.base_model(x)
        return x