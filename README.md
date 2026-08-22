# Prosperity Prognosticator

## Startup Success Prediction Using Machine Learning

Prosperity Prognosticator is a machine learning-based web application that predicts the potential outcome of a startup based on various business and funding-related factors.

The application uses a Random Forest Classification model trained on startup data to predict whether a startup is likely to be **acquired** or **closed**. A Flask web application provides an interactive interface through which users can enter startup details and receive a prediction.

---

## Features

- Startup outcome prediction
- Random Forest machine learning model
- Exploratory Data Analysis (EDA)
- Data preprocessing and missing-value handling
- Categorical feature encoding
- Flask-based web application
- Interactive prediction form
- Trained model saved using Joblib
- Visual analysis of startup data

---

## Technologies Used

### Programming Language
- Python

### Machine Learning
- Scikit-learn
- Random Forest Classifier
- Label Encoding

### Data Analysis
- Pandas
- NumPy
- Matplotlib
- Seaborn

### Web Development
- Flask
- HTML
- CSS

### Model Deployment
- Joblib

---

## Dataset

The project uses a startup dataset containing information related to startup funding, investment rounds, investors, milestones, relationships, and startup status.

Important features used by the model include:

- State Code
- Total Funding
- Funding Rounds
- Venture Capital (VC)
- Angel Investment
- Round A
- Round B
- Round C
- Round D
- Milestones
- Relationships

### Target Variable

The startup status is converted into a binary classification problem:

```text
Acquired → 1
Closed   → 0
