import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model():
    print("Loading California Housing dataset...")
    # Fetch data automatically
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    target = data.target  # Price in $100,000s

    print("Training the model (this may take a few seconds)...")
    
    # Split data: 80% for training, 20% for testing
    X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=42)

    # Initialize the model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X_train, y_train)

    # Validate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Model Training Complete. Mean Squared Error: {mse:.4f}")

    # Save the model
    joblib.dump(model, 'house_price_model.pkl')
    print("Model saved as 'house_price_model.pkl'")

if __name__ == "__main__":
    train_model()