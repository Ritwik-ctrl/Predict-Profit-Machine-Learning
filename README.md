# Predict-Profit-Machine-Learning
Predict startup profit using "R&amp;D Spend" and "Marketing Spend" from "50_Startups.csv" using three different manually implemented machine learning models.
Firstly, it loads dataset and removes missing values while selecting only numerical columns for modeling (R&D Spend, Marketing Spend, Profit) and applies manual feature scaling (standardization).
It then randomly splits the data into 80% training and 20% testing using NumPy.

Models used:
1. Linear Regression
- Uses the Normal Equation to calculate weights.
2. Decision Tree Regressor:
- Recursively splits data based on the feature that minimizes MSE(Mean Squared Error)
3. Support Vector Regression
- Simplified linear SVR.

Evaluation Metrics:
1) R² (Coefficient of Determination) – The higher the better.
2) MAE (Mean Absolute Error) – The lower the better.
3) RMSE (Root Mean Squared Error) – The lower the better.
