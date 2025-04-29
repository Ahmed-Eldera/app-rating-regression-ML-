import pandas as pd
# from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier for classification
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib  # For saving the model

# Load the data
train_data = pd.read_csv('./Split-42/train_data.csv')
valid_data = pd.read_csv('./Split-42/valid_data.csv')
test_data = pd.read_csv('./Split-42/test_data.csv')

# Feature columns (excluding the target 'Rating' and any identifier columns)
feature_columns = ['Reviews', 'Size_MB', 'Installs',  'Price',  'Last_Updated_Year',  'Category_Code', 'Genre_Code']
X_train = train_data[feature_columns]
X_valid = valid_data[feature_columns]
X_test = test_data[feature_columns]

# Target column (e.g., 'Rating')
y_train = train_data['Rating']
y_valid = valid_data['Rating']
y_test = test_data['Rating']

# If there are categorical variables (like 'Last_Updated' or 'Android_Ver'), you may need to encode them
# Example for encoding a categorical feature with LabelEncoder (if needed):
# encoder = LabelEncoder()
# X_train['Last_Updated'] = encoder.fit_transform(X_train['Last_Updated'])
# X_valid['Last_Updated'] = encoder.transform(X_valid['Last_Updated'])
# X_test['Last_Updated'] = encoder.transform(X_test['Last_Updated'])

# Train a RandomForestRegressor model (or use another model as needed)
# model = RandomForestRegressor(n_estimators=100, random_state=42)
model = LinearRegression()

model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'neomodel.joblib')

# Predict on the validation set
y_valid_pred = model.predict(X_valid)
y_test_pred = model.predict(X_test)

# Evaluate the model
valid_mse = mean_squared_error(y_valid, y_valid_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print(f'Validation Mean Squared Error: {valid_mse}')
print(f'Test Mean Squared Error: {test_mse}')
