# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
raw_data = pd.read_excel('Used Car Data.xlsx', sheet_name='Sheet 1 - 1 used_cars')

# Inspect the data
print("Column Names:\n", raw_data.columns)
print("\nData Types:\n", raw_data.dtypes)
print("\nMissing Values:\n", raw_data.isnull().sum())
print("\nSample Data:\n", raw_data.head())

# Rename columns for consistency
raw_data.columns = [
    "Brand", "Model", "Model_Year", "Mileage", "Fuel_Type", 
    "Engine_Specs", "Transmission", "Ext_Color", "Int_Color", 
    "Accident_History", "Title_Status", "Price"
]

# Clean Mileage (Handle non-numeric strings)
# Replace invalid strings with NaN and remove any leading/trailing whitespace
raw_data['Mileage'] = (
    raw_data['Mileage']
    .str.replace('mi.', '', regex=False)  # Remove 'mi.' if it exists
    .str.replace(',', '', regex=False)   # Remove commas
    .str.strip()                         # Remove leading/trailing spaces
)
raw_data['Mileage'] = pd.to_numeric(raw_data['Mileage'], errors='coerce')
cleaned_data = raw_data.dropna(subset=['Mileage'])

# Clean Price (convert to numeric)
raw_data['Price'] = pd.to_numeric(raw_data['Price'], errors='coerce')

# Handle missing values
# Drop rows with missing critical values (e.g., Model_Year, Mileage, Price)
cleaned_data = raw_data.dropna(subset=["Model_Year", "Mileage", "Price"])

# Handle categorical variables 
# For example, encoding Fuel_Type
cleaned_data['Fuel_Type'] = cleaned_data['Fuel_Type'].astype('category').cat.codes

# Treat outliers using IQR method
def treat_outliers(column):
    Q1 = cleaned_data[column].quantile(0.25)
    Q3 = cleaned_data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data[column] = np.where(
        cleaned_data[column] < lower_bound, lower_bound,
        np.where(cleaned_data[column] > upper_bound, upper_bound, cleaned_data[column])
    )

treat_outliers('Mileage')
treat_outliers('Price')

# Create derived features (e.g., car age)
current_year = 2024  # Update based on the current year
cleaned_data['Car_Age'] = current_year - cleaned_data['Model_Year']

# Save the cleaned data to a new file (optional)
cleaned_data.to_csv('cleaned_used_car_data.csv', index=False)

# Final inspection
print("\nCleaned Data Sample:\n", cleaned_data.head())
print("\nRemaining Missing Values:\n", cleaned_data.isnull().sum())

# Feature selection for regression
features = cleaned_data[['Mileage', 'Car_Age']]
target = cleaned_data['Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build and train the regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Display results
results = {
    "Train MSE": train_mse,
    "Test MSE": test_mse,
    "Train R²": train_r2,
    "Test R²": test_r2,
    "Model Coefficients": model.coef_,
    "Intercept": model.intercept_
}

# Visualize Price vs. Mileage relationship
plt.scatter(cleaned_data['Mileage'], cleaned_data['Price'], alpha=0.5)
plt.xlabel('Mileage')
plt.ylabel('Price')
plt.title('Scatter Plot: Mileage vs. Price')
plt.show()

# Output results for regression analysis
for key, values in results.items():
    print(f"{key}, {values}")