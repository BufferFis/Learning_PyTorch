# JUST REWRITING SOMETHING I DID IN PAST TO TEST MYSELF


import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from sklearn.preprocessing import LabelEncoder

from PIL import Image

import pandas as pd
import numpy as np
import os

import opendatasets as od
od.download("https://www.kaggle.com/datasets/andrewmvd/animal-faces")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

image_path = []
labels = []

for i in os.listdir("animal-faces/afhq"):
    for label in os.listdir(f"animal-faces/afhq{i}"):
        for image in os.listdir(f"animal-faces/afhq/{i}/{label}"):
            image_path.append(f"/content/animal-faces/afhq/{i}/{label}/{image}")
            labels.append(label)
    
data_df = pd.DataFrame(zip(image_path, labels), columns=["image_path", "labels"])
data_df.head()


train = data_df.sample(frac = 0.7)
test = data_df.drop(train.index)

val = test.sample(0.5)
test = test.drop(val.index)

label_encoder = LabelEncoder()
label_encoder.fit(data_df["labels"])

transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])

class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform = None):
        self.data = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe["labels"])).to(device)

        def __len__(self):
            return self.data.shape[0]
        
        def __getitem__(self, i):
            img_path = self.dataframe.iloc[i, 0]
            label = self.labels[i]

            image = Image.open(img_path).convert("RGB")
        
            if self.transform:
                image = self.transform(image).to(device)

            return image, label
        

train_dataset = CustomImageDataset(train, transforms)
test_dataset = CustomImageDataset(test, transforms)
val_dataset = CustomImageDataset(val, transforms)

lr = 1e-4
batch_size = 4
epochs = 5

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1)

        self.pooling = nn.MaxPool2d(2,2)

        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128*16*16), 128)

        self.output = nn.Linear(128, len(data_df['labels'].unique()))

    def forward(self, x):
        x = self.conv1(x) # -> (32 , 128, 128)  Convulution keeps size same but increases features, 3 was the initial rgb but now its 32 channels
        x = self.pooling(x) # -> (32, 64, 64) Pooling keeps features same but reduces the size by whatever defined, here its devided by 2
        x = self.relu(x)

        x = self.conv2(x) # -> (64, 64, 64)
        x = self.pooling(x) # -> (64, 32, 32)
        x = self.relu(x)

        x = self.conv3(x) # -> (128, 32, 32)
        x = self.pooling(x) # -> (128, 16, 16)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x
    
model = Net().to(device)


summary(model, input_size = (3,128,128))

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = lr)

for epoch in range(epochs):
    total_loss_train = total_loss_val = total_acc_train = total_acc_val = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss_train += loss.item()
        loss.backward()
        optimizer.step()

        train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
        total_acc_train += train_acc

    with torch.no_grad():
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
            outputs = model(val_inputs)
            val_loss = criterion(outputs, val_labels)

            total_loss_val += val_loss.item()

            val_acc = (torch.argmax(outputs, axis = 1) == val_labels).sum().item()
            total_acc_val += val_acc
    
    print(f""" Epoch {epoch + 1}{epochs}, Train Loss: {round(total_loss_train/1000, 4)}
            Train Accuracy: {round(total_acc_train / train_dataset.__len__() * 100, 4)}
            Validation Loss: {round(total_loss_val/1000, 4)}
            Validation Accuracy: {round(total_acc_val / val_dataset.__len__() * 100, 4)}
            """)
        

with torch.no_grad():
    total_loss_test = 0
    total_acc_test = 0
    for inputs, labels in test_loader:
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        acc = (torch.argmax(predictions, axis = 1) == labels).sum().item()
        total_acc_test += acc
        total_loss_test += loss
print(f"Acc score is: {round(total_acc_test / test_dataset.__len__() * 100, 4)}")


