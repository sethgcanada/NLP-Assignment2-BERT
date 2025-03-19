from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Function to train and evaluate logistic regression for a given dataset size
def train_and_evaluate_log_reg(data_subset, test_texts, test_labels, subset_name):
    # Extract texts and labels
    train_texts = data_subset['text']
    train_labels = data_subset['label']

    # Feature Extraction: TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Use 5000 features to balance performance
    train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    test_tfidf = tfidf_vectorizer.transform(test_texts)

    # Logistic Regression Model
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(train_tfidf, train_labels)

    # Make Predictions
    test_preds = log_reg.predict(test_tfidf)

    # Evaluate Performance
    accuracy = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)

    print(f"Logistic Regression Accuracy for {subset_name}: {accuracy:.2%}")
    print(f"Confusion Matrix for {subset_name}:\n{cm}")

    # Plot Confusion Matrix
    def plot_confusion_matrix(cm, title):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    plot_confusion_matrix(cm, f"Confusion Matrix - Logistic Regression ({subset_name})")

# Create 40K and 80K subsets
data_40k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=20000, random_state=42)
).reset_index(drop=True)

data_80k = train_data.groupby('label', group_keys=False).apply(
    lambda x: x.sample(n=40000, random_state=42)
).reset_index(drop=True)

# Evaluate Logistic Regression for 40K and 80K
train_and_evaluate_log_reg(data_40k, test_data['text'], test_data['label'], "40K Dataset")
train_and_evaluate_log_reg(data_80k, test_data['text'], test_data['label'], "80K Dataset")
