
### 1. **`data_analysis.ipynb`**
   - **Purpose**: This notebook focuses on **exploratory data analysis (EDA)** and initial preprocessing of the dataset.
   - **Key Steps**:
     - Loads the training and test datasets.
     - Performs basic data exploration:
       - Displays the first few rows of the dataset.
       - Summarizes the dataset using `.describe()` and checks for missing values.
     - Handles categorical variables by applying **one-hot encoding**.
     - Splits the data into features (`X`) and target (`y`).
     - Splits the dataset into training and validation sets using `train_test_split`.
     - Trains multiple regression models (e.g., Linear Regression, Decision Tree, Random Forest, etc.) and evaluates their performance using metrics like **MAE**, **MSE**, and **R²**.
     - Visualizes the relationship between actual and predicted ratings using a scatter plot.
   - **Output**:
     - Provides insights into the dataset.
     - Evaluates the performance of various models to identify potential candidates for further tuning.

---

### 2. **`data_preprocessing.ipynb`**
   - **Purpose**: This notebook focuses on **data cleaning and preprocessing** to prepare the dataset for model training.
   - **Key Steps**:
     - Loads the training and test datasets.
     - Handles missing values using forward fill (`fillna(method='ffill')`).
     - Identifies non-numeric columns and converts them to numeric using **one-hot encoding**.
     - Splits the data into features (`X`) and target (`y`).
     - Splits the dataset into training and validation sets.
     - Trains multiple models (similar to data_analysis.ipynb) and evaluates their performance.
     - Selects the best model based on **MAE** and trains it on the full dataset.
     - Prepares the test dataset by ensuring it has the same structure as the training dataset (e.g., matching columns).
     - Generates predictions for the test dataset using the best model.
   - **Output**:
     - A cleaned and preprocessed dataset ready for model training.
     - Predictions for the test dataset.

---

### 3. **`model_training.ipynb`**
   - **Purpose**: This notebook focuses on **training and evaluating models** and generating predictions for the test dataset.
   - **Key Steps**:
     - Loads the training and test datasets.
     - Performs data preprocessing:
       - Handles missing values.
       - Encodes categorical variables using one-hot encoding.
       - Ensures all features are numeric.
     - Splits the data into training and validation sets.
     - Trains multiple regression models (e.g., Linear Regression, Decision Tree, Random Forest, etc.) and evaluates their performance using **MAE**, **MSE**, and **R²**.
     - Selects the best model (e.g., Lasso Regression in this case) based on performance metrics.
     - Prepares the test dataset by ensuring it matches the structure of the training dataset.
     - Generates predictions for the test dataset using the best model.
     - Saves the predictions to a CSV file (`SampleSubmission.csv`).
   - **Output**:
     - Performance metrics for various models.
     - Predictions for the test dataset saved in a CSV file.

---

### **How They Work Together**
1. **`data_analysis.ipynb`**:
   - Provides an initial understanding of the dataset through EDA.
   - Helps identify potential preprocessing steps and suitable models for the task.

2. **`data_preprocessing.ipynb`**:
   - Focuses on cleaning and preparing the dataset for model training.
   - Ensures that the training and test datasets are in the correct format for modeling.

3. **`model_training.ipynb`**:
   - Builds on the preprocessing steps from data_preprocessing.ipynb.
   - Trains and evaluates multiple models.
   - Selects the best model and uses it to generate predictions for the test dataset.

---
