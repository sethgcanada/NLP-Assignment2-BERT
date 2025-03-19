import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns

# Load Datasets
train_file = data_read_path  # Replace with the actual training dataset path
test_file = data_read_path2    # Replace with the actual testing dataset path

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# Verify 'label' column exists and check distribution
print("Training dataset label distribution:")
print(train_data['label'].value_counts())

print("Testing dataset label distribution:")
print(test_data['label'].value_counts())

# Balance the dataset if needed
def balance_dataset(data):
    min_count = data['label'].value_counts().min()
    return data.groupby('label').apply(lambda x: x.sample(n=min_count, random_state=42)).reset_index(drop=True)

if train_data['label'].nunique() == 1 or test_data['label'].nunique() == 1:
    print("Imbalanced dataset detected. Balancing...")
    train_data = balance_dataset(train_data)
    test_data = balance_dataset(test_data)

# Tokenizer Setup
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(data, tokenizer):
    input_ids = []
    attention_masks = []

    for review in data['text']:
        encoded = tokenizer.encode_plus(
            review,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        input_ids.append(encoded['input_ids'].squeeze())
        attention_masks.append(encoded['attention_mask'].squeeze())

    return torch.stack(input_ids), torch.stack(attention_masks), torch.tensor(data['label'].tolist())

# Stratified Sampling for 40K and 80K subsets
data_40k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=20000, random_state=42)
).reset_index(drop=True)

data_80k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=40000, random_state=42)
).reset_index(drop=True)

# Preprocess Train and Test Data for 40K
train_inputs_40k, train_masks_40k, train_labels_40k = preprocess_data(data_40k, tokenizer)
test_inputs_40k, test_masks_40k, test_labels_40k = preprocess_data(test_data, tokenizer)

# Preprocess Train and Test Data for 80K
train_inputs_80k, train_masks_80k, train_labels_80k = preprocess_data(data_80k, tokenizer)
test_inputs_80k, test_masks_80k, test_labels_80k = preprocess_data(test_data, tokenizer)

# Create TensorDatasets
train_dataset_40k = TensorDataset(train_inputs_40k, train_masks_40k, train_labels_40k)
train_dataset_80k = TensorDataset(train_inputs_80k, train_masks_80k, train_labels_80k)
test_dataset = TensorDataset(test_inputs_40k, test_masks_40k, test_labels_40k)  # Same test set for both

# Create DataLoaders
train_loader_40k = DataLoader(train_dataset_40k, batch_size=16, shuffle=True)
train_loader_80k = DataLoader(train_dataset_80k, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Model Setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(model.parameters(), lr=5e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Function to Train and Evaluate
def train_and_evaluate(train_loader, test_loader, scenario_name):
    print(f"Starting Training for {scenario_name}...")
    start_time = time.time()
    for epoch in range(3):  # Number of epochs
        model.train()
        for batch in train_loader:
            input_ids, attention_mask, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Training Complete for {scenario_name}! Total Time: {training_time:.2f} seconds")

    # Testing Loop
    print(f"Starting Testing for {scenario_name}...")
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels_batch = tuple(t.to(device) for t in batch)
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).tolist())
            labels.extend(labels_batch.tolist())

    # Accuracy and Confusion Matrix
    accuracy = accuracy_score(labels, preds)
    cm = confusion_matrix(labels, preds)

    print(f"Testing Accuracy for {scenario_name}: {accuracy:.2%}")
    print(f"Confusion Matrix for {scenario_name}:\n{cm}")

    # Plot Confusion Matrix
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    plot_confusion_matrix(cm, f"Confusion Matrix - {scenario_name}")

# Train and Evaluate for 40K
train_and_evaluate(train_loader_40k, test_loader, "40K Dataset")

# Train and Evaluate for 80K
train_and_evaluate(train_loader_80k, test_loader, "80K Dataset")
