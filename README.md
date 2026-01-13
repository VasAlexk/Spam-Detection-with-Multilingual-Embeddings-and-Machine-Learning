# Spam-Detection-with-Multilingual-Embeddings-and-Machine-Learning
# Ταξινόμηση Email (Spam - Non Spam)

Αυτή η εργασία υλοποιήθηκε στο πλαίσιο του μαθήματος **Στατιστικές Μέθοδοι Μηχανικής Μάθησης**. Σκοπός είναι η ανάπτυξη συστημάτων ταξινόμησης emails σε "Spam" και "Non-Spam" χρησιμοποιώντας διάφορους αλγορίθμους μηχανικής μάθησης.

## Περιγραφή Δεδομένων
Χρησιμοποιήθηκε το σύνολο δεδομένων `emails.csv`. Τα δεδομένα διαχωρίστηκαν σε:
- [cite_start]**Σύνολο Εκπαίδευσης:** 2000 δείγματα [cite: 5, 82]
- [cite_start]**Σύνολο Επικύρωσης:** 1000 δείγματα [cite: 5, 82]
- [cite_start]**Σύνολο Ελέγχου:** 2728 δείγματα [cite: 5, 82]

## Μεθοδολογία & Μοντέλα
Υλοποιήθηκαν και συγκρίθηκαν οι παρακάτω προσεγγίσεις:

1. [cite_start]**Naive Bayes:** Με χρήση αναπαράστασης κειμένου TF-IDF[cite: 33].
2. [cite_start]**k-Nearest Neighbors (k-NN):** Με χρήση **Sentence Embeddings** (μοντέλο `paraphrase-multilingual-MiniLM-L12-v2`)[cite: 81, 89].
3. [cite_start]**Support Vector Machines (SVM):** Δοκιμάστηκαν γραμμικός, πολυωνυμικός και RBF πυρήνας[cite: 158].
4. [cite_start]**PCA & SVM:** Μείωση διάστασης με PCA (διατήρηση 90%, 95%, 99% μεταβλητότητας)[cite: 211, 212].
5. [cite_start]**Logistic Regression:** Εφαρμογή σε 10 διαστάσεις μετά από PCA[cite: 221, 222].

## Αποτελέσματα
[cite_start]Τα μοντέλα αξιολογήθηκαν με βάση τις μετρικές Precision, Recall, F1-score και AUC[cite: 28, 272].

| Μοντέλο | Precision | Recall | F1-score | AUC |
| :--- | :--- | :--- | :--- | :--- |
| Naive Bayes (TF-IDF) | 0.993 | 0.907 | 0.948 | 0.999 |
| k-NN (k=3) | 0.983 | 0.844 | 0.908 | 0.968 |
| SVM (RBF) | 0.962 | 0.945 | 0.953 | 0.998 |
| PCA 95% + SVM | - | - | 0.961 | 0.996 |
| Logistic Regression (PCA=10) | 0.913 | 0.909 | 0.911 | 0.991 |

[cite_start][cite: 274]

[cite_start]Το **SVM με RBF πυρήνα** και **PCA (95%)** παρουσίασε την καλύτερη συνολική απόδοση[cite: 275].

