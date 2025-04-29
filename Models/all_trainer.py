import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

# Load the data
train_data = pd.read_csv('../Split-42/train_data.csv')
valid_data = pd.read_csv('../Split-42/valid_data.csv')
test_data = pd.read_csv('../Split-42/test_data.csv')

# Feature columns (excluding the target 'Rating' and any identifier columns)
feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code' ]
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]

# Target column
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# Define models to try
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42,max_depth=None,min_samples_split=10,min_samples_leaf=2),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42,max_depth=5,min_samples_split=2,subsample=0.8),
    'KNN': KNeighborsRegressor(    n_neighbors=100,
    weights='distance',           # Give closer neighbors more influence
    # algorithm='auto',             # Let sklearn choose the best algorithm
    # leaf_size=30,
    p=1,                          # Use Euclidean distance
    # metric='minkowski',
    n_jobs=-1  ),
    'SVR': SVR(kernel='rbf')
}

# Train, predict, evaluate, and save each model
for name, model in models.items():
    print(f"\nTraining model: {name}")
    model.fit(X_train, y_train)

    y_valid_pred = model.predict(X_valid)
    y_test_pred = model.predict(X_test)

    valid_mse = mean_squared_error(y_valid, y_valid_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"{name} - Validation MSE: {valid_mse:.4f}")
    print(f"{name} - Test MSE: {test_mse:.4f}")

    # Save the model
    joblib.dump(model, f'{name}_model.joblib')
# gp -> Best Parameters: {'model__learning_rate': 0.05, 'model__max_depth': 5, 'model__min_samples_split': 2, 'model__n_estimators': 100, 'model__subsample': 0.8}
#knn -> Best Parameters: {'model__n_neighbors': 100, 'model__p': 1, 'model__weights': 'distance'}