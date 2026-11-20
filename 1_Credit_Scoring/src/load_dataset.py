import pandas as pd
import os

# Dataset location
file_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "data",
    "german.data"
)

# Column names
columns = [
    "checking_account", "duration", "credit_history",
    "purpose", "credit_amount", "savings",
    "employment", "installment_rate", "personal_status",
    "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing",
    "existing_credits", "job", "people_liable",
    "telephone", "foreign_worker", "credit_risk"
]

# Load dataset
data = pd.read_csv(
    file_path,
    sep=r"\s+",
    header=None,
    names=columns
)

# Display dataset
print("CREDIT SCORING DATASET")
print("======================")

print("\nFirst 5 Records:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nColumn Names:")
print(data.columns.tolist())

# STEP 3: DATASET EXPLORATION

print("\n========== DATASET INFORMATION ==========")

# Display dataset information
print("\nDataset Information:")
data.info()

# Check missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check duplicate records
print("\nDuplicate Records:")
print(data.duplicated().sum())

# Statistical description
print("\nStatistical Description:")
print(data.describe())

# Credit Risk Distribution
print("\nCredit Risk Distribution:")
print(data["credit_risk"].value_counts())

# Percentage of Credit Risk
print("\nCredit Risk Percentage:")
print(data["credit_risk"].value_counts(normalize=True) * 100)