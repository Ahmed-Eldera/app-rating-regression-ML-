import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

# Load the datasets
train_data = pd.read_csv('../Split-42/train_data.csv')
valid_data = pd.read_csv('../Split-42/valid_data.csv')
test_data = pd.read_csv('../Split-42/test_data.csv')

# Features and target
feature_columns = ['Reviews', 'Size_MB','Installs', 'Price', 'Last_Updated_Year', 'Category_Code' , 'Version' , 'Min_Android']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Handle missing values and scale in a pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('model', KNeighborsRegressor())
])

# Define hyperparameter grid
param_grid = {
    'model__n_neighbors': [5, 10, 20, 50, 100, 150, 200, 300, 350],
    'model__weights': ['uniform', 'distance'],
    'model__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'model__leaf_size': [20, 30, 40, 50, 60],
    'model__p': [1, 2],  # 1 = Manhattan distance, 2 = Euclidean distance
    'model__metric': ['minkowski', 'euclidean', 'manhattan', 'chebyshev'],
    # 'model__metric_params': [None],  # Optional, usually you can leave this as None
    # 'model__n_jobs': [-1, 1]  # Use all cores (-1) or no parallelism (1)
}


# Perform Grid Search with 5-fold cross-validation
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

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate on validation and test sets
y_valid_pred = best_model.predict(X_valid)
y_test_pred = best_model.predict(X_test)

valid_mse = mean_squared_error(y_valid, y_valid_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# Save the best model
joblib.dump(best_model, 'best_knn_model.joblib')

# Print results
print("Best Parameters:", grid_search.best_params_)
print(f"Validation MSE: {valid_mse}")
print(f"Test MSE: {test_mse}")
