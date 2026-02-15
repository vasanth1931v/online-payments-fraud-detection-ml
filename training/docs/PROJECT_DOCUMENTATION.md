📘 Online Payments Fraud Detection using Machine Learning
📌 1. Introduction

Online Payments Fraud Detection is a machine learning based system designed to identify fraudulent transactions in real-time. The system analyzes transaction features and predicts whether a transaction is legitimate or fraudulent.

📌 2. Technologies Used

Python

Pandas

NumPy

Scikit-learn

XGBoost

Flask

HTML

Bootstrap

GitHub

📌 3. Phase 1 – Data Collection

Dataset Source:
Kaggle – Online Payments Fraud Detection Dataset

Dataset file:
PS_20174392719_1491204439457_log.csv

Steps:

Downloaded dataset

Stored inside project directory

Loaded using pandas

df = pd.read_csv("PS_20174392719_1491204439457_log.csv")

📌 4. Phase 2 – Data Analysis & Visualization

Performed:

Checked dataset shape

Examined columns

Fraud distribution analysis

Correlation analysis

📌 5. Phase 3 – Data Preprocessing

Steps:

Removed unnecessary columns:

nameOrig

nameDest

isFlaggedFraud

Label encoded categorical column (type)

Split dataset into features (X) and target (y)

Train-test split (80%-20%)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

📌 6. Phase 4 – Model Building

Algorithms Applied:

Random Forest

Decision Tree

Extra Trees

SVM

XGBoost

Best Performing Model:
Random Forest (with class balancing)

Evaluation Metrics:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

Model Accuracy Achieved:
~99%

📌 7. Phase 5 – Model Saving

Saved trained model using pickle:

pickle.dump(model, open("payments.pkl", "wb"))

📌 8. Phase 6 – Application Building (Flask)

Created:

home.html

predict.html

submit.html

Flask routes:

/

/predict

/pred

Model integrated using:

model = pickle.load(open("payments.pkl", "rb"))

📌 9. Phase 7 – Running the Application

Command:

cd flask
python app.py


Access via:

http://127.0.0.1:5000

📌 10. Test Cases
Legitimate Transaction
Step: 1
Type: 3
Amount: 9.19
OldbalanceOrg: 170136
NewbalanceOrig: 160296
OldbalanceDest: 0
NewbalanceDest: 0


Output:
Legitimate Transaction

Fraud Transaction
Step: 94
Type: 1
Amount: 500000
OldbalanceOrg: 500000
NewbalanceOrig: 0
OldbalanceDest: 0
NewbalanceDest: 0


Output:
Fraud Detected

📌 11. Deployment

Project pushed to GitHub

Demo video uploaded to SmartBridge

GitHub Repository:
https://github.com/vasanth1931v/online-payments-fraud-detection-ml

📌 12. Conclusion

The project successfully detects fraudulent online transactions using machine learning and provides a user-friendly web interface for prediction.