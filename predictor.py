import pandas as pd
import joblib

# === Load the test data ===
test_data = pd.read_csv('cleaned_test_data.csv')

# === Define the features (must match the training features) ===
feature_columns = ['Reviews', 'Size_MB', 'Installs', 'Price', 'Last_Updated_Year', 'Category_Code']
X_test = test_data[feature_columns]
X_test = X_test.fillna(0)
# === Load your trained model ===
model = joblib.load('./Models/KNN_model.joblib')  # or 'RandomForest_model.joblib', etc.

# === Make predictions ===
test_data['Y'] = model.predict(X_test)

# === Save to a new CSV file ===
test_data.to_csv('SampleSubmission.csv', index=False)

print("Predictions saved to 'test_data_with_predictions.csv'")
