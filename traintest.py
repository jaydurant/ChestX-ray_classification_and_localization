import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import copy
import argparse
import time
import torchvision.transforms as transforms
from models.dataset import XrayDataset, XrayStratifiedDataset
from torch.utils.data import DataLoader
from construct_labels import generate_test_val_train_datasets, generate_stratified_test_val_train_datasets
from models.resnet50 import Resnet50
from models.vgg16 import VGG16
from models.densenet import DenseNet
from models.selected_labels import selected_labels
from utils.metrics import calculate_metrics

parser = argparse.ArgumentParser(description='PyTorch X-ray classification model run')

parser.add_argument("--is_training", type=int, default=1)
parser.add_argument("--epochs",type=int, default=20)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--pretrained", type=int,default=0)
parser.add_argument("--load_model", type=str, default="")
parser.add_argument("--save_model", type=str, default="./saved_models/newmodel.pth")
parser.add_argument("--lr", type=float, default=6.58e-4)
parser.add_argument("--model", type=str, default="resnet")

args = parser.parse_args()
print(args)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

trainx, trainy, testx, testy, valx, valy = generate_stratified_test_val_train_datasets("./padchest_openi_labels.csv")

train_transform = transforms.Compose([
    #transforms.RandomAdjustSharpness(sharpness_factor=0.75),
    transforms.RandomRotation(degrees=(0,180)),
    transforms.RandomInvert(),
    transforms.ToTensor()
])

testval_transform = transforms.Compose([
    transforms.ToTensor()
])

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr


train_dataset = XrayStratifiedDataset("./data", trainx, trainy, train_transform)
val_dataset = XrayStratifiedDataset("./data", valx, valy, testval_transform)
test_dataset = XrayStratifiedDataset("./data", testx, testy, testval_transform)

trainloader = DataLoader(dataset=train_dataset, batch_size = BATCH_SIZE)
valloader = DataLoader(dataset=val_dataset, batch_size = BATCH_SIZE)
testloader = DataLoader(dataset=test_dataset, batch_size = BATCH_SIZE)

print(len(train_dataset), "train")
print(len(val_dataset), "val")
print(len(test_dataset), "test")


num_classes = len(selected_labels)

#access model
model = None
params = None
if args.model == "resnet":
    resnet_model = Resnet50(num_classes)
    resnet_model.to(device)

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
    model = resnet_model
#end feature extractor

if args.model == "vgg":
    vgg_model = VGG16(num_classes)
    vgg_model.to(device)

    for param in vgg_model.parameters():
        param.requires_grad = False
    
    for name, param in vgg_model.named_parameters():
        if "features.28" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True
    
    params = list(vgg_model.classifier.parameters()) + list(vgg_model.features[28].parameters())
    model = vgg_model

if args.model == "desenet": 
    dense_model = DenseNet(num_classes)
    dense_model.to(device)


    for param in dense_model.parameters():
        param.requires_grad = False
    
    for name, param in dense_model.named_parameters():
        if "features.denseblock4.denselayer16.conv2" in name:
            param.requires_grad = True
        if "features.norm5" in name:
            param.requires_grad = True
        if "classifier" in name:
            param.requires_grad = True
    params = list(dense_model.classifier.parameters()) + list(dense_model.features.norm5.parameters()) + list(dense_model.features.denseblock4.denselayer16.conv2.parameters())
    model = dense_model


criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params, lr=LR, weight_decay=1e-4)

#function used for early stopping
def checkstop(arr):
  stop = False
  prev = arr[0]
  for num in arr:
    if num < prev:
      return False
    prev = num
  return True

def runtrainval(model, criterion, optimizer, epochs, trainloader, valloader, path="./saved_modelsls", patience=2, iters=100):
    start = time.time()
    best_val_weights = copy.deepcopy(model.state_dict())
    loss_best = 1000.0
    train_loss_epoch = 0.0
    prev_loss = 1000.0
    prev_loss_arr = []
    earlystopcount = patience + 1

    for epoch in range(epochs):
        training_loss = 0.0
        val_loss = 0.0
        val_batch_count = 0.0
        train_batch_count = 0.0
        image_count = 0.0

        #train
        model.train()
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            train_batch_count += 1.0
            
            train_loss_epoch += loss.item()
            training_loss += loss.item()

            if i % iters  == iters - 1:
                print("Epoch {} Train Step {} - Loss: {} ".format(epoch + 1, i + 1 , training_loss / iters))
                training_loss = 0.0

        print("Epoch {} Train  Final - Loss: {} ".format(epoch + 1, training_loss / train_batch_count))       
        #run val
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valloader):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.float())
                val_loss += loss.item()
                val_batch_count += 1

            val_loss_epoch = val_loss / val_batch_count
            print("Epoch {} Val - Loss{}".format(epoch + 1, val_loss_epoch))
        
        if val_loss_epoch >= prev_loss:
            prev_loss = val_loss_epoch
            prev_loss_arr.append(prev_loss)

            if epoch != 0 and len(prev_loss_arr) >= earlystopcount and checkstop(prev_loss_arr[-earlystopcount:]):
                time_to_train = time.time() - start
                print("Training Time {}min {}sec".format(time_to_train // 60, time_to_train % 60))
                print("Best Validation Loss Achieved {}".format(loss_best))
                return model
            
        else:
            prev_loss = val_loss_epoch
            prev_loss = val_loss_epoch
            prev_loss_arr.append(prev_loss)

        if val_loss_epoch < loss_best:
            loss_best = val_loss_epoch
            best_val_weights = copy.deepcopy(model.state_dict())
            torch.save(best_val_weights, path)
    time_to_train = time.time() - start
    print("Training Time {}min {}sec".format(time_to_train // 60, time_to_train % 60))
    print("Best Validation Loss Achieved {}".format(loss_best))

    model.load_state_dict(best_val_weights)

    return model  


def runtest(model, criterion, testloader, iters):
    predict_arr = []
    truth_arr = []
    test_loss = 0.0
    start = time.time()
    total_batches = 0.0
    model.eval()
    print("start test loss")
    with torch.no_grad():
        for i, data in enumerate(testloader):
            total_batches += 1
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels.float())
            test_loss += loss
            sig = nn.Sigmoid()
            probs = sig(outputs)
            if i == 0:
                print(probs)
            predict_arr.extend(probs.cpu().numpy())
            truth_arr.extend(labels.cpu().numpy())

            if i % iters  == iters - 1:
                print("Test Iter {}".format(i + 1))

    print("Test Loss {}".format(test_loss / total_batches))
    time_to_train = time.time() - start
    print("Training Time {}min {}sec".format(time_to_train // 60, time_to_train % 60))
    metrics = calculate_metrics(np.array(predict_arr), np.array(truth_arr))
    print(metrics)

if args.is_training:
    print("start training")
    runtrainval(model, criterion, optimizer, EPOCHS, trainloader, valloader, path=args.save_model)
    print("finished training")
else:
    print("start test")
    runtest(model, criterion, testloader, iters=100)
    print("finished test")

print("finished")