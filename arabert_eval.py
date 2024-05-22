from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import requests
from torcheval.metrics.functional import multiclass_f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Don't forget to cite the authors for using their dataset for our project.
# Load dataset for training into a pandas dataframe
datasetTrainFilename = "OSACT2022-sharedTask-train-en.csv"
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
datasetTestFilename = "OSACT2022-sharedTask-test-tweets-en.csv"
datasetTestPath = os.path.join(os.getcwd(), datasetTestFilename)
datasetTestInDataframe = pd.read_csv(datasetTestPath, names=['ID', 'Text'], header=None)

# Load labels which are separately stored
labelHSFilename = "OSACT2022-sharedTask-test-HS-gold-labels.csv"
labelHSPath = os.path.join(os.getcwd(), labelHSFilename)
labelHSDataframe = pd.read_csv(labelHSPath, names=['HSOrNot'], header=None)
datasetTestInDataframe.insert(2, "HSOrNot", list(labelHSDataframe['HSOrNot']))
labelOFFFilename = "OSACT2022-sharedTask-test-OFF-gold-labels.csv"
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
datasetDevFilename = "OSACT2022-sharedTask-dev-en.csv"
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
# Loading the tokenizer and model
model_name = "aubmindlab/bert-large-arabertv02-twitter"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

preTrainedModelFilename = "arabic-weights.pth"
modelPath = os.path.join(os.getcwd(), preTrainedModelFilename)

model.load_state_dict(torch.load(modelPath))
tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()

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

predictedLabels = []
trueLabels = []

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

        # Update for f1 score
        predictedLabels = predictedLabels + predicted.tolist()
        trueLabels = trueLabels + labels.tolist()

# Calculate average validation loss
avg_val_loss = val_loss / len(val_loader.dataset)

# Calculate validation accuracy
val_accuracy = val_correct / len(val_loader.dataset)

# Get the f1 score
f1_score = multiclass_f1_score(torch.Tensor(predictedLabels), torch.Tensor(trueLabels), num_classes=2)

print(f"Validation Loss: {avg_val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation F1-Score: {f1_score:.4f}")
