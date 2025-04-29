import pandas as pd
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import joblib

# Load the data
train_data = pd.read_csv('./Split-42/train_data.csv')
valid_data = pd.read_csv('./Split-42/valid_data.csv')

feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code', 'Genre_Code']
X_train = train_data[feature_columns]
y_train = train_data['Rating']
X_valid = valid_data[feature_columns]
y_valid = valid_data['Rating']

# Fill NaNs with 0
imputer = SimpleImputer(strategy='constant', fill_value=0)
X_train = imputer.fit_transform(X_train)
X_valid = imputer.transform(X_valid)

# Pipeline: scale + SVR
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR())
])

# Define hyperparameter space
param_distributions = {
    'svr__kernel': ['poly'],
    'svr__C': [0.1, 1, 10, 100],
    'svr__epsilon': [0.01, 0.1, 0.5, 1.0],
    'svr__gamma': ['scale', 'auto']
}

# RandomizedSearchCV setup
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=15,  # Try 15 random combos
    scoring='neg_mean_squared_error',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Run search
random_search.fit(X_train, y_train)

# Best model
best_model = random_search.best_estimator_
print("Best Parameters:", random_search.best_params_)

# Evaluate
y_pred = best_model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print("Validation MSE:", mse)

# Save model
joblib.dump(best_model, 'best_svr_model.joblib')
