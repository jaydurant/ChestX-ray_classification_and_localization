import torch.nn as nn
import torchvision.models as models

class Resnet50(nn.Module):

    def __init__(self, n_classes):
        super(Resnet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        self.layer4 = resnet.layer4
        self.fc = resnet.fc
        self.base_model = resnet

    def forward(self, x):
        x =self.base_model(x) 
        return x