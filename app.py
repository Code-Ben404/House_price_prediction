import numpy as np
import joblib
import os
from flask import Flask, render_template, request

app = Flask(__name__)

# --- CRITICAL FIX: ABSOLUTE PATH TO MODEL ---
# This ensures Python finds the file regardless of where the server starts
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'house_price_model.pkl')

print(f"Looking for model at: {model_path}") # Debug log for server

if not os.path.exists(model_path):
    print("CRITICAL ERROR: Model file not found on server!")
else:
    print("Model file found.")

try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
# -------------------------------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        try:
            # Get features
            features = [
                float(request.form['OverallQual']),
                float(request.form['GrLivArea']),
                float(request.form['GarageCars']),
                float(request.form['TotalBsmtSF']),
                float(request.form['FullBath']),
                float(request.form['YearBuilt'])
            ]
            
            final_features = [np.array(features)]
            prediction = model.predict(final_features)
            output = round(prediction[0], 2)
            prediction_text = f"${output:,.2f}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
