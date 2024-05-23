from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
from torch.utils.data import DataLoader
import requests
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
import os

# Don't forget to cite the authors for using their dataset for our project.
# Load dataset for training into a pandas dataframe
datasetTrainFilename = "translated-data/OSACT2022-sharedTask-train-en.csv"
datasetTrainPath = os.path.join(os.getcwd(), datasetTrainFilename)
datasetTrainInDataframe = pd.read_csv(datasetTrainPath, names=['ID', 'Text', 'OFFOrNot', 'HSOrNot', 'VLGOrNot', 'VIOOrNot'], header=None)

# Process labels to binary format
accumulatorList = []
for index, row in datasetTrainInDataframe.iterrows():
    entryList = [row['OFFOrNot'][0:3], row['HSOrNot'][0:3], row['VLGOrNot'][0:3], row['VIOOrNot'][0:3]]
    if not all([True if string == "NOT" else False for string in entryList]):
        accumulatorList.append(1)
    else:
        accumulatorList.append(0)

datasetTrainInDataframe.insert(6, "BinLabel", accumulatorList)
datasetTrain = datasetTrainInDataframe[["Text", "BinLabel"]]


# Load test dataset into a pandas dataframe
datasetTestFilename = "translated-data/OSACT2022-sharedTask-test-tweets-en.csv"
datasetTestPath = os.path.join(os.getcwd(), datasetTestFilename)
datasetTestInDataframe = pd.read_csv(datasetTestPath, names=['ID', 'Text'], header=None)

# Load labels which are separately stored
labelHSFilename = "translated-data/OSACT2022-sharedTask-test-HS-gold-labels.csv"
labelHSPath = os.path.join(os.getcwd(), labelHSFilename)
labelHSDataframe = pd.read_csv(labelHSPath, names=['HSOrNot'], header=None)
datasetTestInDataframe.insert(2, "HSOrNot", list(labelHSDataframe['HSOrNot']))
labelOFFFilename = "translated-data/OSACT2022-sharedTask-test-OFF-gold-labels.csv"
labelOFFPath = os.path.join(os.getcwd(), labelOFFFilename)
labelOFFDataframe = pd.read_csv(labelOFFPath, names=['OFFOrNot'], header=None)
datasetTestInDataframe.insert(3, "OFFOrNot", list(labelOFFDataframe['OFFOrNot']))

# Process labels to binary format
accumulatorList = []
for index, row in datasetTestInDataframe.iterrows():
    entryList = [row['OFFOrNot'][0:3], row['HSOrNot'][0:3]]
    if not all([True if string == "NOT" else False for string in entryList]):
        accumulatorList.append(1)
    else:
        accumulatorList.append(0)

datasetTestInDataframe.insert(4, "BinLabel", accumulatorList)
datasetTest = datasetTestInDataframe[["Text", "BinLabel"]]

# Load dev dataset into a pandas dataframe
datasetDevFilename = "translated-data/OSACT2022-sharedTask-dev-en.csv"
datasetDevPath = os.path.join(os.getcwd(), datasetDevFilename)
datasetDevInDataframe = pd.read_csv(datasetDevPath, names=['ID', 'Text', 'OFFOrNot', 'HSOrNot', 'VLGOrNot', 'VIOOrNot'], header=None)

# Process labels to binary format
accumulatorList = []
for index, row in datasetDevInDataframe.iterrows():
    entryList = [row['OFFOrNot'][0:3], row['HSOrNot'][0:3], row['VLGOrNot'][0:3], row['VIOOrNot'][0:3]]
    if not all([True if string == "NOT" else False for string in entryList]):
        accumulatorList.append(1)
    else:
        accumulatorList.append(0)

datasetDevInDataframe.insert(6, "BinLabel", accumulatorList)
datasetDev = datasetDevInDataframe[["Text", "BinLabel"]]

# Next step would be tokenization of the text as input for our model.
modelFilename = "trained_model.pth"
modelPath = os.path.join(os.getcwd(), modelFilename)
model = RobertaForSequenceClassification.from_pretrained("roberta-base")
model.load_state_dict(torch.load(modelPath))
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", use_fast=False)

# Define dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }
    
# Load dataset for training
train_texts = datasetTrain["Text"].tolist()
train_labels = datasetTrain["BinLabel"].tolist()

# Define training dataset and data loader
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Freeze pre-trained model layers
for param in model.roberta.parameters():
   param.requires_grad = False

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=2e-5)

# train loader contains 1029 batches

# Training loop
num_epochs = 35

model.train()
for epoch in range(num_epochs):
    count = 0
    for batch in train_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        print("Epoch: ", epoch, "  Count: ", count)
        count += 1

# Save fineturned model
torch.save(model.state_dict(), "roberta_trained_model_engl_arab.pth")

# Load Datasets
val_texts = datasetTest["Text"].tolist()
val_labels = datasetTest["BinLabel"].tolist()

# Define validation dataset and data loader
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Evaluation loop
model.eval()
val_loss = 0.0
val_correct = 0
counter1 = []


for batch in val_loader:
    counter1.append("0")


print(len(counter1))

counter = 0

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        val_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs.logits, 1)
        val_correct += (predicted == labels).sum().item()
        print("counter", counter)
        counter = counter + 1

# Calculate average validation loss
avg_val_loss = val_loss / len(val_loader.dataset)

# Calculate validation accuracy
val_accuracy = val_correct / len(val_loader.dataset)

print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
