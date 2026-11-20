
import os
import joblib
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# Set project paths
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

data_path = os.path.join(BASE_DIR, "outputs", "processed_data.npz")
model_path = os.path.join(BASE_DIR, "models", "decision_tree.pkl")

# Load processed dataset
data = np.load(data_path)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

print("Processed Dataset Loaded Successfully!")

# Create Decision Tree model
model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

print("\nDecision Tree Model Trained Successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate ROC-AUC score
y_probability = model.predict_proba(X_test)[:, 1]

roc_score = roc_auc_score(y_test, y_probability)

print("ROC-AUC Score:", round(roc_score, 4))

# Save trained model
joblib.dump(model, model_path)

print("\nDecision Tree Model Saved Successfully!")
print("Location:", model_path)
