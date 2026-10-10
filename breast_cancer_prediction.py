# ============================================
# BREAST CANCER PREDICTION USING MACHINE LEARNING
# (Real Wisconsin Diagnostic dataset built into scikit-learn)
# ============================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # saves plots to files; remove this line to show windows
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

# 1. LOAD DATA -------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target          # in this dataset: 0 = malignant, 1 = benign
feature_names = data.feature_names
class_names = data.target_names        # ['malignant' 'benign']

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y
print("Dataset shape:", X.shape)
print("Class counts:\n", df["target"].map(dict(enumerate(class_names))).value_counts())
print("Missing values:", df.isnull().sum().sum())

# 2. TRAIN / TEST SPLIT ---------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print("\nTraining data:", X_train.shape, "| Testing data:", X_test.shape)

# 3. TRAIN MODEL ----------------------------------------------------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 4. EVALUATE -------------------------------------------------------
pred = model.predict(X_test)
print("\nTest accuracy:", round(accuracy_score(y_test, pred), 4))

cv_scores = cross_val_score(model, X, y, cv=5)
print("5-fold CV accuracy: %.4f (+/- %.4f)" % (cv_scores.mean(), cv_scores.std()))

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, pred, target_names=class_names))

# 5. CONFUSION MATRIX PLOT -----------------------------------------
cm = confusion_matrix(y_test, pred)
ConfusionMatrixDisplay(cm, display_labels=class_names).plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

# 6. FEATURE IMPORTANCE --------------------------------------------
imp = pd.Series(model.feature_importances_, index=feature_names).sort_values()
imp.tail(10).plot(kind="barh", figsize=(8, 5), title="Top 10 Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("Top 5 features:\n", imp.sort_values(ascending=False).head(5).round(4))

# 7. PREDICT A NEW SAMPLE (taken from the test set) ----------------
sample = X_test[0].reshape(1, -1)
p = model.predict(sample)[0]
proba = model.predict_proba(sample)[0]
print("\nSample prediction:", class_names[p])
print("Confidence:", dict(zip(class_names, proba.round(3))))
print("Actual label:", class_names[y_test[0]])

# 8. SAVE MODEL -----------------------------------------------------
joblib.dump(model, "breast_cancer_model.joblib")
print("\nModel saved to breast_cancer_model.joblib")
print("This is an educational project, not a clinical diagnosis tool.")
