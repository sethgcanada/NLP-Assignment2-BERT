# Imports, yes I do my imports like this
import os
import time
import torch
import pandas as pd
import seaborn as sns
from google.colab import drive
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# This is the google drive code I used within Google Colab
drive.mount('/content/drive')
data_read_path = os.path.join("drive", "My Drive", "dataset", "train_amazon.csv")
data_read_path2 = os.path.join("drive", "My Drive", "dataset", "test_amazon.csv")

train_data = pd.read_csv(data_read_path)
test_data = pd.read_csv(data_read_path2)


# This is the start of the BERT stuff
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(data, tokenizer):
    ids = []
    masks = []

    for text in data['text']:
        encoded = tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        ids.append(encoded['input_ids'].squeeze())
        masks.append(encoded['attention_mask'].squeeze())

    return torch.stack(ids), torch.stack(masks), torch.tensor(data['label'].tolist())

data40k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=20000, random_state=42)
).reset_index(drop=True)

data80k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=40000, random_state=42)
).reset_index(drop=True)

inputs40k, masks40k, labels40k = preprocess_data(data40k, tokenizer)
tinputs40k, tmasks40k, tlabels40k = preprocess_data(test_data, tokenizer)
inputs80k, masks80k, labels80k = preprocess_data(data80k, tokenizer)
tinputs80k, tmasks80k, tlabels80k = preprocess_data(test_data, tokenizer)

dataset40k = TensorDataset(inputs40k, masks40k, labels40k)
dataset80k = TensorDataset(inputs80k, masks80k, labels80k)
newdataset = TensorDataset(tinputs40k, tmasks40k, tlabels40k)

loader40k = DataLoader(dataset40k, batch_size=16, shuffle=True)
loader80k = DataLoader(dataset80k, batch_size=16, shuffle=True)
testlaoder = DataLoader(newdataset, batch_size=16)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

# I have no idea what this does, but was recommended to add this
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def train_and_evaluate(train_loader, testlaoder, test):
    print(f"Starting Training for {test}...")
    start_time = time.time()
    for epoch in range(3):
        model.train()
        for batch in train_loader:
            ids, mask, labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            outputs = model(ids, attention_mask=mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Complete for {test}! Total Time: {training_time:.2f} seconds")

    print(f"Starting Testing for {test}...")
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in testlaoder:
            ids, mask, labels = tuple(t.to(device) for t in batch)
            outputs = model(ids, attention_mask=mask)
            predictions.extend(torch.argmax(outputs.logits, dim=1).tolist())
            labels.extend(labels.tolist())

    accuracy = accuracy_score(labels, predictions)
    confusion_matrix = confusion_matrix(labels, predictions)

    print(f"Testing Accuracy for {test}: {accuracy:.2%}")
    print(f"Confusion Matrix for {test}:\n{confusion_matrix}")

    def plot_confusion_matrix(confusion_matrix, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    plot_confusion_matrix(confusion_matrix, f"Confusion Matrix - {test}")

train_and_evaluate(loader40k, testlaoder, "40K Dataset")

train_and_evaluate(loader80k, testlaoder, "80K Dataset")


# This is the start of the logistic regression stuff
def train_and_evaluate_log_reg(data_subset, test_texts, test_labels, subset_name):
    train_texts = data_subset['text']
    train_labels = data_subset['label']

    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    test_tfidf = tfidf_vectorizer.transform(test_texts)

    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(train_tfidf, train_labels)

    test_predictions = lr.predict(test_tfidf)

    accuracy = accuracy_score(test_labels, test_predictions)
    confusion_matrix = confusion_matrix(test_labels, test_predictions)

    print(f"Logistic Regression Accuracy for {subset_name}: {accuracy:.2%}")
    print(f"Confusion Matrix for {subset_name}:\n{confusion_matrix}")

    def plot_confusion_matrix(confusion_matrix, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    plot_confusion_matrix(confusion_matrix, f"Confusion Matrix - Logistic Regression ({subset_name})")

data40k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=20000, random_state=42)
).reset_index(drop=True)

data80k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=40000, random_state=42)
).reset_index(drop=True)

train_and_evaluate_log_reg(data40k, test_data['text'], test_data['label'], "40K Dataset")
train_and_evaluate_log_reg(data80k, test_data['text'], test_data['label'], "80K Dataset")