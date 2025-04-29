import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Load datasets
train_data = pd.read_csv('../Split-42/train_data.csv')
valid_data = pd.read_csv('../Split-42/valid_data.csv')
test_data = pd.read_csv('../Split-42/test_data.csv')

# Features and target
feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code', 'Genre_Code']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Define pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('model', RandomForestRegressor(random_state=42))
])

# Define hyperparameter grid
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4],
    'model__max_features': ['sqrt', 'log2',None]
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

# Fit model
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate on validation and test sets
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

valid_mse = mean_squared_error(y_valid, y_valid_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Save the best model
joblib.dump(best_model, 'best_rf_model.joblib')

# Print results
print("Best Parameters:", grid_search.best_params_)
print(f"Validation MSE: {valid_mse}")
print(f"Test MSE: {test_mse}")
#Best Parameters: {'model__max_depth': None, 'model__max_features': 'sqrt', 'model__min_samples_leaf': 2, 'model__min_samples_split': 10, 'model__n_estimators': 100}