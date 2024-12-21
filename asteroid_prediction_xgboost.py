import numpy as np

from scipy.stats import uniform, randint
import pandas as pd


from sklearn.model_selection import  KFold, RandomizedSearchCV

import xgboost as xgb

# Load the dataset
path = "C:\\Users\\Ege Deniz\\Documents\\dataset.csv"
dtype_spec = {'albedo': float, 'H': float, 'diameter': float, 'e': float, 'a': float, 'q': float, 'i': float, 'n': float}
df = pd.read_csv(path, dtype=dtype_spec, low_memory=False)
df = df[['albedo', 'H', 'diameter', 'e', 'a', 'q', 'i', 'n']]
df.fillna(df.median(), inplace=True)



X = df[['albedo', 'H', 'e', 'a', 'q', 'i', 'n']].values
y = df['diameter'].values

# Define the parameter grid
param_dist = {
    'n_estimators': randint(100, 1000),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.5, 1),
    'max_depth': randint(3, 10),
    'colsample_bytree': uniform(0.5, 1),
    'min_child_weight': randint(1, 10)
}

# Initialize the model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Set up the randomized search with cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=100, scoring='neg_mean_squared_error', cv=kfold, random_state=42, n_jobs=-1)

# Fit the model
random_search.fit(X, y)

# Print the results
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: ", -random_search.best_score_)


