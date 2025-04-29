import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Load the datasets
train_data = pd.read_csv('./Split-42/train_data.csv')
valid_data = pd.read_csv('./Split-42/valid_data.csv')
test_data = pd.read_csv('./Split-42/test_data.csv')

# Features and target
feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code', 'Genre_Code']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Pipeline with imputer and model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('model', GradientBoostingRegressor(random_state=1))
])

# Hyperparameter grid for Gradient Boosting
param_grid = {
    'model__n_estimators': [50, 100, 150],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__max_depth': [3, 5, 7],
    'model__subsample': [0.8, 1.0],
    'model__min_samples_split': [2, 5],
}

# Grid Search
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predict
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

# Evaluation
valid_mse = mean_squared_error(y_valid, y_valid_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Save model
joblib.dump(best_model, 'best_gb_model.joblib')

# Print results
print("Best Parameters:", grid_search.best_params_)
print(f"Validation MSE: {valid_mse}")
print(f"Test MSE: {test_mse}")
