import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Loading dataset
df = pd.read_csv("data/bank.csv", sep=';')

df['y'] = df['y'].map({'yes':1, 'no':0})
df = pd.get_dummies(df, drop_first=True)

X = df.drop("y", axis=1)
y = df["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Saving test data for download
X_test_df = X_test.copy()
X_test_df['y'] = y_test.values
X_test_df.to_csv("data/test.csv", index=False)

# Create models folder
if not os.path.exists("models"):
    os.makedirs("models")

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature columns (VERY IMPORTANT for Streamlit)
with open("models/feature_columns.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# Logistic Regression
lr = LogisticRegression(
    max_iter=2000,
    class_weight='balanced'
)
lr.fit(X_train_scaled, y_train)
with open("models/logistic.pkl", "wb") as f:
    pickle.dump(lr, f)

# Decision Tree
dt = DecisionTreeClassifier(
    class_weight='balanced',
    max_depth=8,
    min_samples_leaf=10,
    random_state=42
)
dt.fit(X_train, y_train)
with open("models/decision_tree.pkl", "wb") as f:
    pickle.dump(dt, f)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
with open("models/knn.pkl", "wb") as f:
    pickle.dump(knn, f)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
with open("models/naive_bayes.pkl", "wb") as f:
    pickle.dump(nb, f)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf.fit(X_train, y_train)
with open("models/random_forest.pkl", "wb") as f:
    pickle.dump(rf, f)

# XGBoost
scale = (len(y_train) - sum(y_train)) / sum(y_train)
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42,
    scale_pos_weight=scale
)
xgb.fit(X_train, y_train)
with open("models/xgboost.pkl", "wb") as f:
    pickle.dump(xgb, f)

print("All models trained and saved successfully!")
