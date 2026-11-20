# Credit Scoring Model Using Machine Learning

## 1. Project Overview

The Credit Scoring Model is a machine learning project that predicts whether a person has good or bad credit risk based on their financial information.

The project uses machine learning classification algorithms to analyse customer data and evaluate credit risk.

## 2. Objective

- Predict customer credit risk.
- Classify customers into good and bad credit risk categories.
- Train and evaluate machine learning models.
- Compare the performance of different classification algorithms.

## 3. Dataset

**Dataset Name:** German Credit Data

**Source:** UCI Machine Learning Repository

**Dataset Link:** https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data

Dataset details:
- Total Records: 1000
- Total Features: 20
- Target Column: credit_risk
- Good Credit Risk: 0
- Bad Credit Risk: 1

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## 5. Machine Learning Algorithms

The following classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier

## 6. Data Preprocessing

The following preprocessing techniques are used:

- Dataset loading and exploration.
- Missing value and duplicate checking.
- Separation of features and target.
- Label mapping for credit risk.
- Standard Scaling for numerical features.
- One-Hot Encoding for categorical features.
- Train-test splitting (80% training and 20% testing).

## 7. Model Performance

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 78% | 0.804 |
| Decision Tree | 65% | 0.6741 |

**Best Performing Model:** Logistic Regression

## 8. Project Structure

```text
1_Credit_Scoring/
│
├── data/
│   └── german.data
│
├── models/
│   ├── preprocessor.pkl
│   ├── logistic_regression.pkl
│   └── decision_tree.pkl
│
├── outputs/
│   ├── processed_data.npz
│   └── model_comparison.png
│
├── src/
│   ├── load_dataset.py
│   ├── preprocess_data.py
│   ├── logistic_regression.py
│   ├── decision_tree.py
│   └── model_comparison.py
│
├── README.md
└── requirements.txt
```

## 9. How to Run the Project

### Step 1: Install required libraries

```bash
pip install -r 1_Credit_Scoring/requirements.txt
```

### Step 2: Load the dataset

```bash
python 1_Credit_Scoring/src/load_dataset.py
```

### Step 3: Preprocess the dataset

```bash
python 1_Credit_Scoring/src/preprocess_data.py
```

### Step 4: Train Logistic Regression

```bash
python 1_Credit_Scoring/src/logistic_regression.py
```

### Step 5: Train Decision Tree

```bash
python 1_Credit_Scoring/src/decision_tree.py
```

### Step 6: Compare Model Performance

```bash
python 1_Credit_Scoring/src/model_comparison.py
```

## 10. Conclusion

The project demonstrates how machine learning classification algorithms can be used for credit risk prediction.

Two algorithms were implemented and evaluated. Logistic Regression achieved 78% accuracy, while Decision Tree achieved 65% accuracy.

Based on the evaluation results, Logistic Regression performed better on this dataset.

**Note:** This is an educational machine learning project and should not be used as the sole basis for real-world lending decisions.