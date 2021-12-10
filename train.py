import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
from models.dataset import XrayDataset
from torch.utils.data import DataLoader
from construct_labels import generate_test_val_train_datasets
from models.resnet50 import Resnet50
from models.selected_labels import selected_labels

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train, val, test = generate_test_val_train_datasets("./padchest_img_labels.csv")


train_transform = transforms.Compose([
    #transforms.RandomAdjustSharpness(sharpness_factor=0.75),
    transforms.RandomRotation(degrees=(0,180)),
    transforms.RandomInvert(),
    transforms.ToTensor()
])

testval_transform = transforms.Compose([
    transforms.PILToTensor()
])


EPOCHS = 20
BATCH_SIZE = 10
LR = 1e-3


train_dataset = XrayDataset("./data", train, train_transform)
val_dataset = XrayDataset("./data", val, testval_transform)
test_dataset = XrayDataset("./data", test, testval_transform)

trainloader = DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE)
valloader = DataLoader(dataset=val_dataset, batch_size = BATCH_SIZE)
testloader = DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE)

print(len(train_dataset), "train")
print(len(val_dataset), "val")
print(len(test_dataset), "test")


num_classes = len(selected_labels)

#access model
resnet_model = Resnet50(num_classes)

#use as feature extractor turn in function
for param in resnet_model.parameters():
    param.requires_grad = False

for name, param in resnet_model.named_parameters():
    if "layer4.2.conv3" in name:
        param.requires_grad = True
    if "layer4.2.bn3" in name:
        param.requires_grad = True
    if "fc" in name:
        param.requires_grad = True

params = list(resnet_model.fc.parameters()) + list(resnet_model.layer4[2].bn3.parameters()) + list(resnet_model.layer4[2].conv3.parameters())
#end feature extractor


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params, lr=LR)

def train(model, criterion, optimizer, epochs, trainloader, valloader):

    for epoch in range(epochs):
        training_loss = 0.0
        val_loss = 0.0

        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            training_loss += loss.item()

            if i % 2 == 1:
                print("Epoch {} Train - Loss: {} ".format(epoch + 1, training_loss / 2))
                training_loss = 0.0

print("start training")
train(resnet_model, criterion, optimizer, EPOCHS, trainloader, valloader)
print("finished training")




