import os
import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Logistic Regression", "Decision Tree"]

# Evaluation results
accuracy = [78.0, 65.0]
roc_auc = [0.804, 0.6741]

# Set bar positions
x = np.arange(len(models))
width = 0.35

# Convert ROC-AUC into percentage for comparison
roc_auc_percentage = [score * 100 for score in roc_auc]

# Create bar chart
plt.figure(figsize=(9, 5))

plt.bar(x - width/2, accuracy, width, label="Accuracy")
plt.bar(x + width/2, roc_auc_percentage, width, label="ROC-AUC")

plt.xlabel("Machine Learning Models")
plt.ylabel("Score (%)")
plt.title("Credit Scoring Model Performance Comparison")

plt.xticks(x, models)
plt.ylim(0, 100)
plt.legend()

plt.tight_layout()

# Save chart
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
output_path = os.path.join(BASE_DIR, "outputs", "model_comparison.png")

plt.savefig(output_path)
plt.show()

print("Model Comparison Chart Saved Successfully!")
print("Location:", output_path)