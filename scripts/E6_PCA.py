import pandas as pd
import re
import string
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score

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

# To καλύτερο kernel από Ερώτημα 5
best_kernel = "rbf" 

print(" PCA + SVM results ")
for var in [0.90, 0.95, 0.99]:
    pca = PCA(n_components=var, random_state=42)
    X_train_pca = pca.fit_transform(X_train_emb)
    X_val_pca   = pca.transform(X_val_emb)
    X_test_pca  = pca.transform(X_test_emb)

    svm = SVC(kernel=best_kernel, probability=True)
    svm.fit(X_train_pca, y_train)

    y_val_pred = svm.predict(X_val_pca)
    y_val_prob = svm.predict_proba(X_val_pca)[:, 1]

    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    auc = roc_auc_score(y_val, y_val_prob)

    print(f"Variance={int(var*100)}% | PCA dims={X_train_pca.shape[1]} | Val F1={f1:.4f} | Val AUC={auc:.4f}")
