# 🛡️ Insurance Fraud Detection using Machine Learning

## 📌 Project Overview

Insurance fraud is a major challenge for insurance companies, leading to significant financial losses every year. The objective of this project is to build an end-to-end Machine Learning model capable of identifying potentially fraudulent insurance claims based on customer, policy, vehicle, and incident-related information.

The project covers the complete Machine Learning lifecycle, including data preprocessing, exploratory data analysis (EDA), feature engineering, model building, hyperparameter tuning, evaluation, and deployment using Flask and AWS.

---

# 🎯 Problem Statement

The goal of this project is to predict whether an insurance claim is fraudulent or genuine.

Target Variable:

- fraud_reported
    - 1 → Fraudulent Claim
    - 0 → Genuine Claim

Since missing a fraudulent claim is more costly than investigating a genuine claim, Recall was considered the primary evaluation metric during model selection.

---

# 📂 Dataset

Dataset Source:
https://www.kaggle.com/datasets/arpan129/insurance-fraud-detection

Dataset contains information related to

- Customer Details
- Insurance Policy Information
- Incident Information
- Vehicle Information
- Claim Information

Number of records: 1000

Target Classes

- Fraud
- Not Fraud

---

# 📊 Exploratory Data Analysis (EDA)

## Univariate Analysis

Performed distribution analysis for

- Numerical Features
- Categorical Features

Visualizations

- Histogram
- Box Plot
- Count Plot
- Percentage Distribution

Checked

- Missing Values
- Outliers
- Class Distribution

---

## Bivariate Analysis

Studied relationship between

- Feature vs Target

Visualizations

- Count Plot
- Stacked Percentage Plot
- Box Plot
- Violin Plot

Performed

- Numerical vs Target
- Categorical vs Target

---

## Multivariate Analysis

Performed

- Correlation Heatmap
- Pairwise Relationship Analysis
- Feature Correlation

Detected highly correlated features.

---

# 🧹 Data Cleaning

Performed the following preprocessing

- Removed duplicate records
- Checked missing values
- Replaced '?' values
- Extracted useful date features
- Removed unnecessary columns

Dropped Columns

- policy_number
- incident_location
- insured_zip
- total_claim_amount

Reason

- policy_number → Unique Identifier
- incident_location → High Cardinality
- insured_zip → High Cardinality
- total_claim_amount → Derived from
    - injury_claim
    - property_claim
    - vehicle_claim

---

# ⚙️ Feature Engineering

Extracted

From policy_bind_date

- policy_bind_year
- policy_bind_month
- policy_bind_day

Dropped

- policy_bind_date

From incident_date

- incident_month
- incident_day

Dropped

- incident_date

Year was ignored because all incidents occurred in 2015.

---

# 🔄 Data Preprocessing

Implemented using ColumnTransformer.

### One-Hot Encoding

Applied on

- policy_state
- policy_csl
- insured_occupation
- insured_hobbies
- insured_relationship
- incident_type
- collision_type
- incident_severity
- authorities_contacted
- incident_state
- incident_city
- auto_make
- auto_model

---

### Binary Encoding

Applied on

- insured_sex
- property_damage
- police_report_available

---

### Ordinal Encoding

Applied on

insured_education_level

Education order

- High School
- Associate
- College
- Bachelor
- Masters
- MD
- PhD

---

### Numerical Scaling

Applied StandardScaler on numerical features.

---

# ✂️ Train-Test Split

Training Data

80%

Testing Data

20%

The preprocessing pipeline was fitted only on the training data to prevent data leakage.

---

# 🤖 Machine Learning Models

The following classification models were trained.

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- K-Nearest Neighbors
- AdaBoost
- Gradient Boosting
- CatBoost
- Linear Support Vector Classifier
- Gaussian Naive Bayes

---

# 📈 Evaluation Metrics

Models were evaluated using

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Since this is a fraud detection problem, Recall was considered the primary metric.

Reason

False Negative

Fraud predicted as genuine.

Financial loss for insurance company.

False Positive

Genuine claim predicted as fraud.

Can be manually investigated.

Hence maximizing Recall is more important than maximizing Accuracy.

---

# 🔍 Hyperparameter Tuning

Performed using

RandomizedSearchCV

Cross Validation

5-Fold Cross Validation

Scoring Metric

Recall

---

# 🏆 Best Performing Model

Decision Tree Classifier

Performance

- Accuracy : 85%
- Precision : 64.6%
- Recall : 85.7%
- F1 Score : 73.7%
- ROC-AUC : 84%

The tuned Decision Tree significantly reduced overfitting and achieved the highest Recall among all models, making it the most suitable model for fraud detection.

---

# 📉 ROC-AUC Interpretation

ROC Curve represents

True Positive Rate (Recall)

vs

False Positive Rate

AUC Score obtained

0.84

Interpretation

If one fraudulent claim and one genuine claim are selected randomly, the model has approximately an 84% probability of assigning a higher fraud score to the fraudulent claim than to the genuine claim.

---

# 📦 Model Deployment

The final trained model and preprocessing pipeline will be serialized using

- Pickle
or
- Joblib

Deployment Stack

- Flask
- HTML
- CSS
- Docker
- GitHub Actions
- AWS ECR
- AWS EC2

Workflow

User Input

↓

Flask Application

↓

Preprocessing Pipeline

↓

Decision Tree Model

↓

Prediction

↓

Fraud / Not Fraud

---

# 📁 Project Structure

```
Insurance-Fraud-Detection
│
├── notebook
│
├── artifacts
│
├── data
│
├── templates
│
├── static
│
├── src
│
├── pipeline
│
├── app.py
│
├── requirements.txt
│
├── Dockerfile
│
├── README.md
│
└── setup.py
```

---

# 🚀 Technologies Used

Programming Language

- Python

Libraries

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- CatBoost

Deployment

- Flask
- Docker
- GitHub Actions
- AWS EC2
- AWS ECR

---

# 📌 Future Improvements

- SHAP Explainability
- Feature Importance Dashboard
- Probability-based Risk Score
- Model Monitoring
- Drift Detection
- CI/CD Automation
- MLflow Experiment Tracking

---

# 📚 Learning Outcomes

This project demonstrates an end-to-end Machine Learning workflow including

- Data Cleaning
- Feature Engineering
- Exploratory Data Analysis
- Data Preprocessing
- Feature Encoding
- Pipeline Creation
- Model Building
- Hyperparameter Tuning
- Model Evaluation
- ROC-AUC Analysis
- Model Serialization
- Flask Deployment
- Docker Containerization
- CI/CD Pipeline
- AWS Deployment

---

# 👨‍💻 Author

Srimannarayana

Machine Learning | Data Science | MLOps