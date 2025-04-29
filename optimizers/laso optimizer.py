import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import joblib

# Load the datasets
train_data = pd.read_csv('../Split-42/train_data.csv')
valid_data = pd.read_csv('../Split-42/valid_data.csv')
test_data = pd.read_csv('../Split-42/test_data.csv')

# Define the feature columns and target
feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code', 'Genre_Code']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Pipeline: handle missing values and define model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('model', Lasso(max_iter=10000))
])

# Define the hyperparameter grid
param_grid = {
    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
}

# Grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on validation and test sets
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

valid_mse = mean_squared_error(y_valid, y_valid_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Save the best model
joblib.dump(best_model, 'best_lasso_model.joblib')

# Output results
print("Best Parameters:", grid_search.best_params_)
print(f"Validation MSE: {valid_mse}")
print(f"Test MSE: {test_mse}")
