# Churn-EDA-and-Prediction
The goal of this project is to understand customer behavior and predict churn (whether a customer will stop using a service) using a real-world telecom dataset. This can help businesses retain valuable customers and take preventive actions..

🔍 EDA Highlights
Handled missing values and categorized features like Age and Tenure.
Explored relationships between churn and features like Support Calls, Contract Length, etc.
Visualized trends using matplotlib.

🤖 Modeling
Model: RandomForestClassifier
Preprocessing: OneHotEncoding with ColumnTransformer, handling NaNs, reshaping labels
Evaluation: Accuracy score on train-test split
Future: Add XGBoost, feature importance, and clustering

Requirements
pandas
numpy
matplotlib
scikit-learn
xgboost

