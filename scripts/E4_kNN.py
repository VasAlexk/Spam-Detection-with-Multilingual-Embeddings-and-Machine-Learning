import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
X_val_emb   = model.encode(X_val_clean.tolist(), show_progress_bar=True)
X_test_emb  = model.encode(X_test_clean.tolist(), show_progress_bar=True)

# k gia validation
k_list = [1, 3, 5, 7, 9, 11, 15]
best_k = None
best_f1 = -1

print(" Validation results ")
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_emb, y_train)

    y_val_pred = knn.predict(X_val_emb)

    y_val_prob = knn.predict_proba(X_val_emb)[:, 1]

    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_val_prob)

    print(f"k={k:>2} | F1={f1:.4f} | AUC={auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_k = k

print(f"\nBest k on validation (by F1): {best_k} (F1={best_f1:.4f})")

# teliko training
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train_emb, y_train)

y_pred = knn.predict(X_test_emb)
y_prob = knn.predict_proba(X_test_emb)[:, 1]

print("\n Best k-NN ")
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1       :", f1_score(y_test, y_pred, zero_division=0))
print("AUC      :", roc_auc_score(y_test, y_prob))

#confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Spam", "Spam"]
)

disp.plot(cmap=plt.cm.Blues)
plt.title(f"Confusion Matrix - k-NN (k={best_k})")
plt.show()

#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(
    fpr,
    tpr,
    label=f"k-NN (k={best_k}, AUC = {roc_auc_score(y_test, y_prob):.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - k-NN")
plt.legend()
plt.show()