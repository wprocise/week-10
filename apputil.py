# apputil.py file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Exercise #1
# 1) Load the coffee analysis data from a CSV file
data = pd.read_csv("https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv")

# 2) Split the data into training and testing sets
"""Split the data into training and testing sets with 80:20 ratio.
    linear regression model predicts 'rating' based on '100g USD' feature.
"""
df_train, df_test = train_test_split(data, test_size=0.2)
features = ['100g_USD']
X_train = df_train[features]
y_train = df_train['rating']
lm = LinearRegression()
lm.fit(X_train.values, y_train.values)

# 3) Saved trained model in this repository as a pickle file
# 3) Save the trained model in this repository as a pickle file called 'model_1.pickle'
"""Save the trained model in this repository as a pickle file called 'model_1.pickle'."""
with open('model_1.pickle', 'wb') as f:
    pickle.dump(lm, f)

