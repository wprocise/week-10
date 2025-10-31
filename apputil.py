# apputil.py file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Exercise #1
# 1) Load the coffee analysis data from a CSV file
data = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

# 2)
"""Split the data into training and testing sets with 80:20 ratio.
    linear regression model predicts 'rating' based on '100g USD' feature.
"""
df_train, df_test = train_test_split(data, test_size=0.2)
features = ['100g_USD']
X_train = df_train[features]
y_train = df_train['rating']
lm = LinearRegression()
lm.fit(X_train.values, y_train.values)

# 3)
"""Save the trained model in this repository as a pickle file called 'model_1.pickle'."""
with open('model_1.pickle', 'wb') as f:
    pickle.dump(lm, f)

# Exercise #2
# Updating script from exercise #1
"""Encode categorical 'roast column into numerical labels.
    Prepare features and target.
    Split data into training and testing sets (80:20).
    Train Decision Tree Regressor Model to predict 'rating'.
    Save trained model as 'model_2.pickle'.
"""
# Training Decision Tree Regressor Model
roast_cat = {cat: idx for idx, cat in enumerate(data['roast'].unique())}
data['roast_num'] = data['roast'].map(roast_cat)

features = ['100g_USD', 'roast_num']
X = data[features]
y = data['rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)




