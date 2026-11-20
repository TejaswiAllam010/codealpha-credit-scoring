import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# STEP 1: Load Dataset

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

file_path = os.path.join(BASE_DIR, "data", "german.data")

columns = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable",
    "telephone", "foreign_worker", "credit_risk"
]

data = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

print("Dataset Loaded Successfully!")

# STEP 2: Separate Features and Target

X = data.drop("credit_risk", axis=1)

# Good risk = 0, Bad risk = 1
y = data["credit_risk"].map({1: 0, 2: 1})

print("\nFeatures and Target Separated!")

# STEP 3: Identify Numerical and Categorical Columns

numerical_columns = X.select_dtypes(include=["number"]).columns

categorical_columns = X.select_dtypes(exclude=["number"]).columns

# STEP 4: Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns)
    ]
)

# STEP 5: Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# STEP 6: Transform Data

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

# STEP 7: Create Output Folders

output_dir = os.path.join(BASE_DIR, "outputs")
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(output_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# STEP 8: Save Processed Data

np.savez_compressed(
    os.path.join(output_dir, "processed_data.npz"),
    X_train=X_train_processed,
    X_test=X_test_processed,
    y_train=y_train.to_numpy(),
    y_test=y_test.to_numpy()
)

# Save Preprocessor

joblib.dump(
    preprocessor,
    os.path.join(model_dir, "preprocessor.pkl")
)

# STEP 9: Display Results

print("\n========== PREPROCESSING RESULTS ==========")

print("Original Dataset:", data.shape)

print("Training Records:", X_train_processed.shape)

print("Testing Records:", X_test_processed.shape)

print("Training Labels:", y_train.shape)

print("Testing Labels:", y_test.shape)

print("\nPreprocessing Completed Successfully!")

print("Processed data saved in outputs folder.")

print("Preprocessor saved in models folder.")