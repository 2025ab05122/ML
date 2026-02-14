import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# Page Configuration

st.set_page_config(
    page_title="Bank Marketing Classification",
    layout="wide"
)

st.title("Bank Marketing Classification App")
st.markdown(
    """
    This application evaluates multiple machine learning models  
    for predicting whether a customer subscribes to a term deposit.
    """
)
st.markdown("---")

# Load Saved Models

@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": pickle.load(open("models/logistic.pkl", "rb")),
        "Decision Tree": pickle.load(open("models/decision_tree.pkl", "rb")),
        "KNN": pickle.load(open("models/knn.pkl", "rb")),
        "Naive Bayes": pickle.load(open("models/naive_bayes.pkl", "rb")),
        "Random Forest": pickle.load(open("models/random_forest.pkl", "rb")),
        "XGBoost": pickle.load(open("models/xgboost.pkl", "rb")),
    }
    scaler = pickle.load(open("models/scaler.pkl", "rb"))
    feature_columns = pickle.load(open("models/feature_columns.pkl", "rb"))
    return models, scaler, feature_columns

models, scaler, feature_columns = load_models()

# Download Test Dataset

st.subheader("Download Test Dataset")

if os.path.exists("data/test.csv"):
    with open("data/test.csv", "rb") as f:
        st.download_button(
            label="Download Test Data (CSV)",
            data=f,
            file_name="test.csv",
            mime="text/csv"
        )
else:
    st.warning("Test dataset not found. Please regenerate using train.py.")

st.markdown("---")

# Upload Dataset

st.subheader("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing feature columns and target column 'y'",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "y" not in df.columns:
        st.error("Uploaded file must contain target column 'y'.")
        st.stop()

    X = df.drop("y", axis=1)
    y = df["y"]

    # Ensure correct feature alignment
    if y.dtype == object:
        y = y.map({'yes': 1, 'no': 0})

    # Apply one-hot encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Align with training columns
    X_encoded = X_encoded.reindex(columns=feature_columns, fill_value=0)

    X = X_encoded
    st.markdown("---")

    # Model Selection

    st.subheader("Select Model")

    model_name = st.selectbox(
        "Choose a Model",
        list(models.keys())
    )

    model = models[model_name]

    # Apply scaling only when required
    if model_name in ["Logistic Regression", "KNN", "Naive Bayes"]:
        X_input = scaler.transform(X)
    else:
        X_input = X.values

    # Predictions
    y_pred = model.predict(X_input)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_input)[:, 1]
    else:
        y_prob = y_pred

    # Evaluation Metrics

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", f"{precision:.4f}")
    col5.metric("Recall", f"{recall:.4f}")
    col6.metric("F1 Score", f"{f1:.4f}")

    st.markdown("---")

    # Confusion Matrix

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap="Blues",
        ax=ax
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    st.pyplot(fig)

else:
    st.info("Please upload a test CSV file to evaluate a model.")
