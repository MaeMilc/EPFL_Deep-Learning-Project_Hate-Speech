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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer, RobertaModel, RobertaForSequenceClassification
import os


# Don't forget to cite the authors for using their dataset for our project.
# Load the dataset into a pandas dataframe
datasetFilename = "Dynamically_Generated_Hate_Dataset_v0.2.3.csv"
datasetPath = os.path.join(os.getcwd(), datasetFilename)
datasetInDataframe = pd.read_csv(datasetPath)

# Get dataset for training and make textlabels binary
datasetTrain = datasetInDataframe[["text", "label", "split"]]
datasetTrain = datasetTrain[datasetTrain["split"] == "train"]
datasetTrain["label"] = datasetTrain["label"].map({"hate": 1, 'nothate': 0})
datasetTrain = datasetTrain.drop(columns="split")

# Get dataset for testing and make labels binary
datasetTest = datasetInDataframe[["text", "label", "split"]]
datasetTest = datasetTest[datasetTest["split"] == "test"]
datasetTest["label"] = datasetTest["label"].map({"hate": 1, 'nothate': 0})
datasetTest = datasetTest.drop(columns="split")

# Get dev dataset, whatever that is, and make labels binary
datasetDev = datasetInDataframe[["text", "label", "split"]]
datasetDev = datasetDev[datasetDev["split"] == "dev"]
datasetDev["label"] = datasetDev["label"].map({"hate": 1, 'nothate': 0})
datasetDev = datasetDev.drop(columns="split")

# Next step would be tokenization of the text as input for our model.


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = model = RobertaForSequenceClassification.from_pretrained('roberta-base')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

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
train_texts = datasetTrain["text"].tolist()
train_labels = datasetTrain["label"].tolist()

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
num_epochs = 5

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
torch.save(model.state_dict(), "trained_model.pth")


# Load Datasets
val_texts = datasetTest["text"].tolist()
val_labels = datasetTest["label"].tolist()

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
