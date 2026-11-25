# Credit Scoring Model Using Machine Learning

## Project Overview
This project uses Machine Learning algorithms to predict customer credit risk based on financial and personal information. It classifies customers into good credit risk and bad credit risk categories.

Developed as part of the CodeAlpha Machine Learning Internship – Task 1.

## Objectives
- Analyze customer credit data.
- Perform data preprocessing and feature encoding.
- Build machine learning classification models.
- Evaluate and compare model performance.
- Predict customer credit risk.

## Dataset
- Dataset: German Credit Dataset (UCI Machine Learning Repository)
- Total Records: 1,000
- Features: 20 input features
- Target: Credit Risk
- Good Credit Risk: 0
- Bad Credit Risk: 1

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Machine Learning Algorithms
1. Logistic Regression
2. Decision Tree Classifier

## Model Performance

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 78% | 80.4% |
| Decision Tree | 65% | 67.41% |

Best Performing Model: Logistic Regression

## Project Structure

```text
CodeAlpha_CreditScoring/
│
├── 1_Credit_Scoring/
│   ├── data/
│   ├── models/
│   ├── outputs/
│   ├── src/
│   └── requirements.txt
│
└── README.md
```

## Key Features
- Data preprocessing and feature scaling.
- Categorical feature encoding.
- Model training and testing.
- Confusion matrix evaluation.
- Model performance comparison.

## Results
Logistic Regression achieved 78% accuracy and 80.4% ROC-AUC, outperforming the Decision Tree model on the test dataset.

## Disclaimer
This project is developed for educational purposes as part of the CodeAlpha Machine Learning Internship.
