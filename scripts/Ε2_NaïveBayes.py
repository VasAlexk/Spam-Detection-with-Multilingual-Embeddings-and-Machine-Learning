import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# Load dataset και shuffle
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


#Καθαρισμός κειμένου
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


#TF-IDF + Naive Bayes
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train_clean)
X_val_vec   = vectorizer.transform(X_val_clean)
X_test_vec  = vectorizer.transform(X_test_clean)

nb = MultinomialNB()
nb.fit(X_train_vec, y_train)


#Predictions
y_pred = nb.predict(X_test_vec)
y_prob = nb.predict_proba(X_test_vec)[:, 1]

# Metrics
print("Naive Bayes Test Set ")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
print("F1-score :", f1_score(y_test, y_pred, zero_division=0))
print("AUC      :", roc_auc_score(y_test, y_prob))


#Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Spam", "Spam"]
)

disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Naive Bayes")
plt.show()

#report
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Non-Spam", "Spam"],
    zero_division=0
))


# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(
    fpr,
    tpr,
    label=f"Naive Bayes (AUC = {roc_auc_score(y_test, y_prob):.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Naive Bayes")
plt.legend()
plt.show()
