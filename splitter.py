import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
data = pd.read_csv('cleaned_data.csv')

# First split: Train (60%) and temp (40%)
train_data, temp_data = train_test_split(data, test_size=0.8, random_state=50)

# Second split: Split temp data into validation (50%) and test (50%)
valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=50)

# Save the splits to CSV files
train_data.to_csv('./Split-42/train_data.csv', index=False)
valid_data.to_csv('./Split-42/valid_data.csv', index=False)
test_data.to_csv('./Split-42/test_data.csv', index=False)

print("Data has been split and saved into train_data.csv, valid_data.csv, and test_data.csv.")
