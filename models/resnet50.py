import torch.nn as nn
import torchvision.models as models

class Resnet50(nn.Module):

    def __init__(self, n_classes):
        super(Resnet50, self).__init__()
        resnet = models.resnet50(pretrained=True)
        resnet.avgpool = nn.Identity(2048)
        resnet.fc= nn.Sequential(
            nn.Conv2d(2048,2048, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Linear(in_features=2048,out_features=n_classes, bias=True)
        )
        self.base_model = resnet

    def forward(self, x):
        x = self.base_model(x)
        return x