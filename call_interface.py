# Import the necessary libraries for running all code
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
import sys

# Import our scripts handling respective model training, finetuning and evaluation
import arabert_arabic_eval
import bertweet_arabic_translated_english_pretrained
import bertweet_english_base_training
import roberta_arabic_translated_english_pretrained
import roberta_custom_classifier_arabic_translated_english_pretrained
import roberta_custom_classifier_english_base_training
import roberta_english_base_training

# Defining a new classification layer for Roberta such as to improve our results
# Please note that our code is an adapted version of the transformers Roberta Classifier as in https://github.com/huggingface/transformers/blob/84ea427f460ffc8d2ddc08a341ccda076c24fc1f/src/transformers/models/roberta/modeling_roberta.py
class CustomClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.normalize = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        residual = x
        x = self.dense1(x)
        x = torch.tanh(x)
        x = self.normalize(x + residual)
        x = self.dropout(x)
        residual = x
        x = self.dense2(x)
        x = torch.tanh(x)
        x = self.normalize(x + residual)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

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

def evaluate_pretrained(val_texts, val_labels, model, tokenizer, criterion):
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

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: call_interface.py <baseModel> <language> <mode>")
        sys.exit(1)
    
    baseModel = sys.argv[1]
    language = sys.argv[2]
    mode = sys.argv[3]

    # In this case, training will be performed as we've done it (including a single evaluation run afterwards)
    if mode == "train":
        if baseModel == "arabert" and language == "arabic":
            train, test, dev = arabert_arabic_eval.load_dataset()
            mod, tok = arabert_arabic_eval.load_model_tokenizer()
            newMod, crit = arabert_arabic_eval.training(train, tok, mod)
            arabert_arabic_eval.evaluation(test, newMod, tok, crit)
        
        elif baseModel == "bertweet" and language == "translated":
            train, test, dev = bertweet_arabic_translated_english_pretrained.load_dataset()
            mod, tok = bertweet_arabic_translated_english_pretrained.load_model_tokenizer()
            newMod, crit = bertweet_arabic_translated_english_pretrained.training(train, tok, mod)
            bertweet_arabic_translated_english_pretrained.evaluation(test, newMod, tok, crit)

        elif baseModel == "bertweet" and language == "english":
            train, test, dev = bertweet_english_base_training.load_dataset()
            mod, tok = bertweet_english_base_training.load_model_tokenizer()
            newMod, crit = bertweet_english_base_training.training(train, tok, mod)
            bertweet_english_base_training.evaluation(test, newMod, tok, crit)

        elif baseModel == "roberta" and language == "translated":
            train, test, dev = roberta_arabic_translated_english_pretrained.load_dataset()
            mod, tok = roberta_arabic_translated_english_pretrained.load_model_tokenizer()
            newMod, crit = roberta_arabic_translated_english_pretrained.training(train, tok, mod)
            roberta_arabic_translated_english_pretrained.evaluation(test, newMod, tok, crit)

        elif baseModel == "roberta" and language == "english":
            train, test, dev = roberta_english_base_training.load_dataset()
            mod, tok = roberta_english_base_training.load_model_tokenizer()
            newMod, crit = roberta_english_base_training.training(train, tok, mod)
            roberta_english_base_training.evaluation(test, newMod, tok, crit)

        elif baseModel == "custom" and language == "translated":
            train, test, dev = roberta_custom_classifier_arabic_translated_english_pretrained.load_dataset()
            mod, tok = roberta_custom_classifier_arabic_translated_english_pretrained.load_model_tokenizer()
            newMod, crit = roberta_custom_classifier_arabic_translated_english_pretrained.training(train, tok, mod)
            roberta_custom_classifier_arabic_translated_english_pretrained.evaluation(test, newMod, tok, crit)

        elif baseModel == "custom" and language == "english":
            train, test, dev = roberta_custom_classifier_english_base_training.load_dataset()
            mod, tok = roberta_custom_classifier_english_base_training.load_model_tokenizer()
            newMod, crit = roberta_custom_classifier_english_base_training.training(train, tok, mod)
            roberta_custom_classifier_english_base_training.evaluation(test, newMod, tok, crit)
        
        # In this case, an unsupported argument was passed
        else:
            print("Unsupported argument")
            print("Supported arguments are:")
            print("baseModel can be arabert, bertweet, custom or bertweet")
            print("language can be arabic, english, translated")
            print("mode can be train or eval")
                      
    # In this case, only evaluation will be performed as we've done it
    elif mode == "eval":
        if baseModel == "arabert" and language == "arabic":
            train, test, dev = arabert_arabic_eval.load_dataset()
            mod, tok = arabert_arabic_eval.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["Text"].tolist()
            labels = test["BinLabel"].tolist()

            # Load the pretrained model
            modelFilename = "arabert-weights.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)
        
        elif baseModel == "bertweet" and language == "translated":
            train, test, dev = bertweet_arabic_translated_english_pretrained.load_dataset()
            mod, tok = bertweet_arabic_translated_english_pretrained.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["Text"].tolist()
            labels = test["BinLabel"].tolist()

            # Load the pretrained model
            modelFilename = "bertweet_trained_engl_arab.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)

        elif baseModel == "bertweet" and language == "english":
            train, test, dev = bertweet_english_base_training.load_dataset()
            mod, tok = bertweet_english_base_training.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["text"].tolist()
            labels = test["label"].tolist()

            # Load the pretrained model
            modelFilename = "bertweet_trained_engl.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)

        elif baseModel == "roberta" and language == "translated":
            train, test, dev = roberta_arabic_translated_english_pretrained.load_dataset()
            mod, tok = roberta_arabic_translated_english_pretrained.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["Text"].tolist()
            labels = test["BinLabel"].tolist()

            # Load the pretrained model
            modelFilename = "roberta_trained_model_engl_arab.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)

        elif baseModel == "roberta" and language == "english":
            train, test, dev = roberta_english_base_training.load_dataset()
            mod, tok = roberta_english_base_training.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["text"].tolist()
            labels = test["label"].tolist()

            # Load the pretrained model
            modelFilename = "trained_model.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)

        elif baseModel == "custom" and language == "translated":
            train, test, dev = roberta_custom_classifier_arabic_translated_english_pretrained.load_dataset()
            mod, tok = roberta_custom_classifier_arabic_translated_english_pretrained.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["Text"].tolist()
            labels = test["BinLabel"].tolist()

            # Load the pretrained model
            modelFilename = "custom_classifier_trained_engl_arab.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)

        elif baseModel == "custom" and language == "english":
            train, test, dev = roberta_custom_classifier_english_base_training.load_dataset()
            mod, tok = roberta_custom_classifier_english_base_training.load_model_tokenizer()

            # Prepare the test datasets
            texts = test["text"].tolist()
            labels = test["label"].tolist()

            # Load the pretrained model
            modelFilename = "custom_classifier_trained_engl.pth"
            modelPath = os.path.join(os.getcwd(), modelFilename)
            # Replace the classifier with the custom one
            mod.classifier = CustomClassificationHead(mod.config)
            mod.load_state_dict(torch.load(modelPath))

            # Prepare the criterion used
            crit = nn.CrossEntropyLoss()

            # Perform evaluation
            evaluate_pretrained(texts, labels, mod, tok, crit)
        
        # In this case, an unsupported argument was passed
        else:
            print("Unsupported argument")
            print("Supported arguments are:")
            print("baseModel can be arabert, bertweet, custom or bertweet")
            print("language can be arabic, english, translated")
            print("mode can be train or eval")

    # In this case, an unsupported argument was passed
    else:
        print("Unsupported argument")
        print("Supported arguments are:")
        print("baseModel can be arabert, bertweet, custom or bertweet")
        print("language can be arabic, english, translated")
        print("mode can be train or eval")