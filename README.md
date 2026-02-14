# Bank Marketing Classification

## Problem Statement
The objective of this project is to implement and compare multiple machine learning classification models to predict whether a bank client will subscribe to a term deposit.
This is a binary classification problem where the goal is to predict the target variable:
y = 1 → Client subscribed
y = 0 → Client did not subscribe
The project includes model implementation, evaluation using multiple performance metrics and deployment of an interactive Streamlit web application.

---
## Dataset Description

The dataset used is the **Bank Marketing Dataset**.
It contains customer information collected during direct marketing campaigns conducted by a Portuguese banking institution.

### Key Characteristics:

* Binary classification problem
* 16 input features
* 45,000+ instances
* Mix of numerical and categorical features
* Imbalanced target distribution (~88% No, ~12% Yes)

## Input Features (16 Total)
### Bank Client Data

1. **age** (numeric)  
2. **job** (categorical)  
3. **marital** (categorical)  
4. **education** (categorical)  
5. **default** (binary)  
6. **balance** (numeric)  
7. **housing** (binary)  
8. **loan** (binary)  

### Contact Information (Current Campaign)

9. **contact** (categorical)  
10. **day** (numeric)  
11. **month** (categorical)  
12. **duration** (numeric)  

### Campaign History

13. **campaign** (numeric)  
14. **pdays** (numeric; -1 means not previously contacted)  
15. **previous** (numeric)  
16. **poutcome** (categorical)  

---
## Models Implemented

The following six classification models were implemented and evaluated on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Naive Bayes (Gaussian)
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

Class imbalance was handled using:

* `class_weight='balanced'` for Logistic Regression, Decision Tree, and Random Forest
* `scale_pos_weight` for XGBoost

Feature scaling was applied only where necessary (Logistic Regression, KNN, Naive Bayes).

---

## Model Evaluation Metrics

Each model was evaluated using the following metrics:

* Accuracy
* AUC (Area Under ROC Curve)
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

## Comparison Table

| ML Model Name               | Accuracy | AUC    | Precision | Recall | F1 | MCC    |
| ------------------- | -------- | ------ | --------- | ------ | -------- | ------ |
| Logistic Regression | 0.8460   | 0.9079 | 0.4186    | 0.8138 | 0.5528   | 0.5091 |
| Decision Tree       | 0.8095   | 0.8891 | 0.3663    | 0.8611 | 0.5140   | 0.4777 |
| kNN                 | 0.8936   | 0.8084 | 0.5860    | 0.3091 | 0.4047   | 0.3742 |
| Naive Bayes         | 0.8639   | 0.8088 | 0.4282    | 0.4877 | 0.4560   | 0.3797 |
| Random Forest (Ensemble)      | 0.9041   | 0.9289 | 0.6914    | 0.3261 | 0.4432   | 0.4319 |
| XGBoost (Ensemble)            | 0.8738   | 0.9263 | 0.4767    | 0.8025 | 0.5981   | 0.5542 |

---

## Observations 

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Achieved strong Recall (0.8138) and balanced MCC (0.5091), indicating effective detection of the minority class while maintaining stable overall performance. Demonstrates that linear decision boundaries work reasonably well for this dataset. |
| Decision Tree | Achieved the highest Recall (0.8611), meaning it detects most positive cases. However, lower Precision (0.3663) suggests a higher number of false positives, indicating possible overfitting and reduced generalization compared to ensemble methods. |
| kNN | Achieved relatively high Accuracy (0.8936) but low Recall (0.3091), showing that it performs well on the majority class but struggles to detect minority class instances. Distance-based methods are less effective on imbalanced datasets. |
| Naive Bayes | Produced moderate and balanced performance across metrics. While not the top model in this scenario, it serves as a reliable probabilistic baseline model with consistent results. |
| Random Forest (Ensemble) | Achieved the highest Accuracy (0.9041) and high Precision (0.6914), indicating strong ability to reduce false positives. However, lower Recall (0.3261) suggests conservative prediction of the positive class. |
| XGBoost (Ensemble) | Achieved the highest MCC (0.5542), strong Recall (0.8025), and the highest F1 Score (0.5981). Given the imbalanced dataset, MCC and F1 Score are more informative than Accuracy. Therefore, XGBoost provides the best overall balanced performance. |

---