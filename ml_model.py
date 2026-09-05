"""
Week 4 Task - Machine Learning Model Development and Evaluation
Beginner-level project: Predicting whether a breast tumor is
Malignant or Benign using Logistic Regression.

Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Source: Publicly available inside scikit-learn (sklearn.datasets)
This is a well-known, free, beginner-friendly dataset often used
to practice classification problems.
"""

# -----------------------------
# STEP 1: Import the libraries I need
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# -----------------------------
# STEP 2: Load the dataset
# -----------------------------
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target   # 0 = malignant, 1 = benign

print("Shape of dataset:", df.shape)
print(df.head())

# -----------------------------
# STEP 3: Data cleaning / preprocessing
# -----------------------------
# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum().sum())  # sklearn's built-in dataset has 0 missing values

# Separate features (X) and target/label (y)
X = df.drop("target", axis=1)
y = df["target"]

# Split data into training set and testing set (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling - many ML algorithms perform better when features
# are on a similar scale (mean = 0, standard deviation = 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# STEP 4: Model selection and training
# -----------------------------
# I chose Logistic Regression because:
# 1. This is a binary classification problem (Malignant vs Benign)
# 2. Logistic Regression is simple, fast, and easy for beginners to
#    understand and interpret
# 3. It works well as a first "baseline" model before trying more
#    complex algorithms
model = LogisticRegression(max_iter=5000, random_state=42)
model.fit(X_train_scaled, y_train)

# -----------------------------
# STEP 5: Make predictions
# -----------------------------
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probabilities for ROC curve

# -----------------------------
# STEP 6: Evaluate the model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy of the model:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Malignant", "Benign"]))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# STEP 7: Visualization 1 - Confusion Matrix
# -----------------------------
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = [0, 1]
plt.xticks(tick_marks, ["Malignant", "Benign"])
plt.yticks(tick_marks, ["Malignant", "Benign"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

# Add the numbers inside each box
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                  color="black", fontsize=14)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# -----------------------------
# STEP 8: Visualization 2 - ROC Curve
# -----------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# -----------------------------
# STEP 9: Compare training vs testing accuracy (check overfitting)
# -----------------------------
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print("\nTraining Accuracy:", round(train_accuracy * 100, 2), "%")
print("Testing Accuracy:", round(test_accuracy * 100, 2), "%")

print("\nDone! Visualizations saved as confusion_matrix.png and roc_curve.png")
