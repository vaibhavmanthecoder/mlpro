# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# Load data
calories = pd.read_csv('/calories.csv')
exe = pd.read_csv('/exercise.csv')

# Combine datasets
calories_data = pd.concat([exe, calories['Calories']], axis=1)

# Convert gender to numerical
calories_data.replace({"Gender": {'male': 0, 'female': 1}}, inplace=True)

# Prepare features and target
x = calories_data.drop(columns=['User_ID', 'Calories'])
y = calories_data['Calories']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train model
model = XGBRegressor()
model_fit = model.fit(x_train, y_train)

# Predict
test_data_prediction = model.predict(x_test)

# Evaluate
mea = metrics.mean_absolute_error(y_test, test_data_prediction)
print(mea)
