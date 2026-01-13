import pandas as pd
import re
import string
import numpy as np

from sentence_transformers import SentenceTransformer

#Load dataset
df = pd.read_csv("emails.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df["text"].astype(str)
y = df["spam"].astype(int)


#Split
X_train = X.iloc[:2000]
y_train = y.iloc[:2000]

X_val = X.iloc[2000:3000]
y_val = y.iloc[2000:3000]

X_test = X.iloc[3000:]
y_test = y.iloc[3000:]

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

print("Embeddings shapes:")
print("Train:", X_train_emb.shape, "Val:", X_val_emb.shape, "Test:", X_test_emb.shape)
