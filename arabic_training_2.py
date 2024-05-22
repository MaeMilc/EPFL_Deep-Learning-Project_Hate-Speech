from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import re

# Loading the tokenizer and model
model_name = "aubmindlab/bert-large-arabertv02-twitter"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def remove_emojis(text):
    # Regex to filter out emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def process_file(file_path, number):
    data = []
    if(number == 1):
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) < 6:
                    continue  # Skip malformed lines
                id, tweet_text, off_label, hs_label, vulgar_label, violence_label = parts
                # Replace "<LF>" with ".", remove emojis
                tweet_text = tweet_text.replace('<LF>', '.')
                tweet_text = remove_emojis(tweet_text)
                data.append([id, tweet_text, off_label, hs_label, vulgar_label, violence_label])
        return pd.DataFrame(data, columns=['id', 'tweet_text', 'OFF_label', 'HS_label', 'Vulgar_label', 'Violence_label'])
    else:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue  # Skip malformed lines
                id, tweet_text = parts
                # Replace "<LF>" with ".", remove emojis
                tweet_text = tweet_text.replace('<LF>', '.')
                tweet_text = remove_emojis(tweet_text)
                data.append([id, tweet_text])
        return pd.DataFrame(data, columns=['id', 'tweet_text'])

# Example usage:
dataset_train = process_file('arabic-data/OSACT2022-sharedTask-train.txt', 1)
dataset_dev = process_file('arabic-data/OSACT2022-sharedTask-dev.txt', 1)
dataset_test = process_file('arabic-data/OSACT2022-sharedTask-test-tweets.txt', 2)

# Saving to CSV
dataset_train.to_csv('arabic-data/OSACT2022-sharedTask-train.csv', index=False)
dataset_dev.to_csv('arabic-data/OSACT2022-sharedTask-dev.csv', index=False)
dataset_test.to_csv('arabic-data/OSACT2022-sharedTask-test-tweets.csv', index=False)

# Define a function to load the dataset and map labels
def load_and_process_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath, delimiter=',')
    
    # Define a mapping function for binary labels
    def map_label(row):
        if 'NOT_' in row['OFF_label'] and 'NOT_' in row['Vulgar_label'] and 'NOT_' in row['Violence_label'] and 'HS' not in row['HS_label']:
            return 0
        return 1
    
    # Apply the mapping function to create a binary label
    df['binary_label'] = df.apply(map_label, axis=1)
    
    # Select and rename necessary columns
    df = df[['tweet_text', 'binary_label']]
    df.columns = ['text', 'label']
    
    return df

# Paths to the datasets
train_path = 'arabic-data/OSACT2022-sharedTask-train.csv'
dev_path = 'arabic-data/OSACT2022-sharedTask-dev.csv'
test_path = 'arabic-data/OSACT2022-sharedTask-test-tweets.csv'


text = "sampleBro"
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
    


# Load and process datasets
dataset_train = load_and_process_data(train_path)
dataset_dev = load_and_process_data(dev_path)
dataset_test = load_dataset('csv', data_files='arabic-data/OSACT2022-sharedTask-test-tweets.csv')


# Load dataset for training
train_texts = dataset_train["text"].tolist()
train_labels = dataset_train["label"].tolist()


train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=128)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# Model training function
def train_model(model, train_loader, optimizer, criterion, epochs=35):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch {epoch+1}: Loss {total_loss/len(train_loader)}')


        
# Training setup
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

# Freeze pre-trained model layers
for param in model.bert.parameters():
    param.requires_grad = False


# Run this cell to train the model
train_model(model, train_loader, optimizer, criterion)


# Save the model and weights
# model.save_pretrained('arabert-pretrained')
torch.save(model.state_dict(), "arabert-weights.pth")


# Load Datasets
val_texts = dataset_dev["text"].tolist()
val_labels = dataset_dev["label"].tolist()

# Define validation dataset and data loader
val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=128)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)



model.eval()
val_loss = 0.0
val_correct = 0

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