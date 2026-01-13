import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve
)

# Load
df = pd.read_csv("emails.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["text"].astype(str)
y = df["spam"].astype(int)

# Split
X_train, y_train = X.iloc[:2000], y.iloc[:2000]
X_val,   y_val   = X.iloc[2000:3000], y.iloc[2000:3000]
X_test,  y_test  = X.iloc[3000:], y.iloc[3000:]

# katharismos
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

X_train_clean = X_train.apply(clean_text)
X_val_clean   = X_val.apply(clean_text)
X_test_clean  = X_test.apply(clean_text)

# Embeddings
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X_train_emb = model.encode(X_train_clean.tolist(), show_progress_bar=True)
X_test_emb  = model.encode(X_test_clean.tolist(), show_progress_bar=True)

# PCA to 10 dims
pca = PCA(n_components=10, random_state=42)
X_train_pca = pca.fit_transform(X_train_emb)
X_test_pca  = pca.transform(X_test_emb)

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_pca, y_train)

y_pred = lr.predict(X_test_pca)
y_prob = lr.predict_proba(X_test_pca)[:, 1]

print("Logistic Regression με PCA=10 Test Set")
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1       :", f1_score(y_test, y_pred, zero_division=0))
print("AUC      :", roc_auc_score(y_test, y_prob))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Spam", "Spam"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Logistic Regression (PCA=10)")
plt.show()

#roc matrix
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (PCA=10, AUC = {roc_auc_score(y_test, y_prob):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression (PCA=10)")
plt.legend()
plt.show()