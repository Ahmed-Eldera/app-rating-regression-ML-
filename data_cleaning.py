import pandas as pd
import numpy as np

# Load your dataset (replace with your actual file path)
df = pd.read_csv('train.csv')

# Rename columns for clarity
df.columns = ['App', 'Category', 'Reviews', 'Size', 'Installs', 'IsFree',
              'Price', 'Age', 'Genre', 'Last_Updated',
              'Current_Ver', 'Android_Ver','Rating']

# Clean 'Size' column (convert to MB)
def size_to_mb(size_str):
    try:
        if 'M' in size_str:
            return float(size_str.replace('M', ''))
        elif 'k' in size_str:
            return float(size_str.replace('k', '')) / 1024
        elif size_str == 'Varies with device':
            return np.nan
    except:
        return np.nan

df['Size_MB'] = df['Size'].apply(size_to_mb)

# Clean 'Installs' (remove '+' and ',')
df['Installs'] = df['Installs'].str.replace('+', '', regex=False).str.replace(',', '', regex=False)
df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')

# Extract year from 'Last_Updated'
df['Last_Updated_Year'] = pd.to_datetime(df['Last_Updated'], errors='coerce').dt.year

# Convert version to float if possible
def version_to_float(version):
    try:
        return float(version.split('.')[0] + '.' + version.split('.')[1])
    except:
        return np.nan

df['Version'] = df['Current_Ver'].apply(version_to_float)

# Clean 'Android_Ver' to float
df['Min_Android'] = df['Android_Ver'].str.extract(r'(\d+\.?\d*)')
df['Min_Android'] = pd.to_numeric(df['Min_Android'], errors='coerce')

# Encode 'Price' as Free/Not Free
df['Is_Free'] = df['IsFree'].apply(lambda x: 1 if x == 'Free' else 0)

# Encode categorical columns
df['Category_Code'] = df['Category'].astype('category').cat.codes
df['Price_Code'] = df['Price'].astype('category').cat.codes
df['Genre_Code'] = df['Genre'].astype('category').cat.codes

# Drop rows with missing critical data
df_cleaned = df.dropna(subset=['Rating', 'Size_MB', 'Installs', 'Last_Updated_Year','Version','Min_Android'])
# df_cleaned = df.dropna(subset=[])

# Save cleaned data to new CSV
df_cleaned.to_csv('cleaned_data.csv', index=False)

print("Cleaned data saved to 'cleaned_data.csv'")
