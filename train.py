# Imports here
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import get_input_arg as parser

args = parser.get_train_parser().parse_args()

data_dir = args.dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

datatransforms = {
    'train': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.RandomHorizontalFlip(p=0.2),
                                transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'test': transforms.Compose([transforms.Resize(255),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'valid': transforms.Compose([transforms.Resize(255),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
}

datasets = {
    'train': datasets.ImageFolder(train_dir, transform=datatransforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=datatransforms['test']),
    'valid': datasets.ImageFolder(valid_dir, transform=datatransforms['valid'])
}

dataloader = {
    'train': torch.utils.data.DataLoader(datasets['train'], batch_size=32, shuffle=True),
    'test': torch.utils.data.DataLoader(datasets['test'], batch_size=32),
    'valid': torch.utils.data.DataLoader(datasets['valid'], batch_size=32)
}


model = eval(f'models.{args.arch}')(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(25088, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1))


model.classifier = classifier
device = 'cuda' if (torch.cuda.is_available() and args.gpu) else 'cpu'
print(device)
model.to(device)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
epochs = args.epochs
train_loss = 0
steps = 0
for_every = 5

for e in range(epochs):
    for images, labels in dataloader['train']:
        
        images, labels = images.to(device), labels.to(device)
        steps += 1
        optimizer.zero_grad()
        log_ps = model.forward(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if steps % for_every == 0:
            model.eval()
            accuracy = 0
            validation_loss = 0
            with torch.no_grad():
                for images, labels in dataloader['valid']:

                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images)
                    loss_val=criterion(log_ps,labels)
                    validation_loss+=loss_val.item()
                    ps = torch.exp(log_ps).to(device)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            print(f"Epoch: {e+1}   Training Loss: {train_loss/for_every}   Validation Loss: {validation_loss/len(dataloader['valid'])}   Validation Accuracy: {accuracy.item()/len(dataloader['valid'])}")
            train_loss = 0
            model.train()
print("Done")

# Saving the checkpoint
model.class_to_idx = datasets['train'].class_to_idx
path = args.save_dir
state_dict = {
    'hidden_units':args.hidden_units,
    'arch':args.arch,
    'optimizer_state_dict': optimizer.state_dict(),
    'model_state_dict': model.state_dict()
}
torch.save(state_dict, path)