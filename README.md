# Transaction-Fraud-detection-app
Overview

This is a web app built with Streamlit that checks whether a bank transaction is likely to be fraud. The user enters the details of a transaction, and the app returns a fraud probability and a clear FRAUD or NORMAL result. It uses a trained XGBoost machine learning model.

How it works
Input: the user enters 7 transaction details: Step (hour since start), Transaction Type (CASH_IN, CASH_OUT, DEBIT, PAYMENT or TRANSFER), Amount, and the sender’s and receiver’s balances before and after the transaction.
Feature engineering: the app builds 9 model features from these inputs: hour of day, night-time flag (hours 0 to 8), sender-account-empty flag, balance changes for the sender and receiver, balance mismatch errors, log of the amount, and the encoded transaction type.
Scaling: the features are scaled with the same MinMaxScaler used during training.
Prediction: the XGBoost model returns the fraud probability, and the app compares it with an adjustable threshold (default 0.50).
Output: it shows the probability, a progress bar, and a red FRAUD or green NORMAL message.
Model and data
Dataset: 30,000 transactions, of which 10% (3,000) are fraud.
Cleaning: the raw data was cleaned in Python (duplicates, inconsistent text, symbols in numbers and missing values), then engineered with extra features.
Algorithm: XGBoost, tuned with GridSearchCV and scored on F1.
Performance: on the 30% hold-out test set, it reaches about 0.99 F1, with precision about 0.99 and recall about 0.98.
Key fraud patterns found: the account being emptied to 0, large CASH_OUT and TRANSFER amounts, and balance mismatches.
Features
A simple form with number inputs and a type dropdown.
A threshold slider: lower catches more fraud but gives more false alarms.
Fast predictions, because the model and scaler are loaded once and cached.
It can be deployed free on Streamlit Community Cloud.
Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Streamlit, Pickle.

Short version for a resume or portfolio

Built and deployed a Streamlit web app that detects fraudulent bank transactions in real time. Cleaned and engineered 30,000 records, trained a GridSearchCV-tuned XGBoost model (about 0.99 F1), and exposed it through an interactive form that returns a fraud probability and an adjustable fraud threshold.

If you’d like, I can turn this into a README.md file for your GitHub repository.
