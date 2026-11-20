import os
import joblib
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# STEP 1: Set project paths

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(
    BASE_DIR, "outputs", "processed_data.npz"
)

model_path = os.path.join(
    BASE_DIR, "models", "logistic_regression.pkl"
)

# STEP 2: Load processed data

data = np.load(data_path)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Processed Dataset Loaded Successfully!")

# STEP 3: Create Logistic Regression Model

model = LogisticRegression(max_iter=1000)

# STEP 4: Train Model

model.fit(X_train, y_train)

print("\nLogistic Regression Model Trained Successfully!")

# STEP 5: Make Predictions

y_pred = model.predict(X_test)

# STEP 6: Evaluate Model

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# STEP 7: ROC-AUC Score

y_probability = model.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_probability)

print("ROC-AUC Score:", round(roc_score, 4))

# STEP 8: Save Model

joblib.dump(model, model_path)

print("\nModel Saved Successfully!")
print("Location:", model_path)