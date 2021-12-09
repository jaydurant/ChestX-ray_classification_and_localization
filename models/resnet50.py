import torch.nn as nn
import torchvision.models as models

class Resnet50(nn.Module):

    def __init__(self, n_classes):
        super(Resnet50, self).__init()

        resnet = models.resnet50(pretrained=True)
        resnet.fc = nn.Sequential(
            nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        )
        self.base_model = resnet
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.sig(self.base_model(x))
        return x