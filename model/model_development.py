import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load Dataset (LOCALLY)
# This looks for 'housing.csv' in the same folder as this script
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'housing.csv')

print(f"Loading dataset from: {file_path}")

if not os.path.exists(file_path):
    print("ERROR: 'housing.csv' not found!")
    print("Please download it and save it in the 'model' folder.")
    exit()

df = pd.read_csv(file_path)

# 2. Feature Selection
required_columns = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
df = df[required_columns]

# 3. Preprocessing
print("Preprocessing data...")
df = df.fillna(df.median())

X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Train Model
print("Training Random Forest Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluation
print("Evaluating Model...")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 7. Save Model
joblib.dump(model, os.path.join(current_dir, 'house_price_model.pkl'))
print("Model saved as 'house_price_model.pkl'")