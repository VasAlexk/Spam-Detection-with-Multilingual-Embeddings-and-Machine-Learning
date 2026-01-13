import pandas as pd
import re
import string
import numpy as np
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve


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

# kernels gia validation
kernels = ["linear", "poly", "rbf"]
best_kernel = None
best_f1 = -1

print(" Validation results")
for k in kernels:
    svm = SVC(kernel=k, probability=True)  # probability=True για AUC
    svm.fit(X_train_emb, y_train)

    y_val_pred = svm.predict(X_val_emb)
    y_val_prob = svm.predict_proba(X_val_emb)[:, 1]

    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_val_prob)

    print(f"kernel={k:<6} | F1={f1:.4f} | AUC={auc:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_kernel = k

print(f"\nBest kernel on validation (by F1): {best_kernel} (F1={best_f1:.4f})")

# Final test evaluation me to kalytero kernel
svm = SVC(kernel=best_kernel, probability=True)
svm.fit(X_train_emb, y_train)

y_pred = svm.predict(X_test_emb)
y_prob = svm.predict_proba(X_test_emb)[:, 1]

print("\n Test results ")
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
plt.title(f"Confusion Matrix - SVM ({best_kernel})")
plt.show()

#roc curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(
    fpr,
    tpr,
    label=f"SVM ({best_kernel}, AUC = {roc_auc_score(y_test, y_prob):.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.legend()
plt.show()