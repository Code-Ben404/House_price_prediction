import sqlite3
import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# Load the trained model
model = joblib.load('house_price_model.pkl')

def init_db():
    """Create the database and table if they don't exist."""
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            med_inc REAL,
            house_age REAL,
            ave_rooms REAL,
            ave_bedrms REAL,
            population REAL,
            ave_occup REAL,
            latitude REAL,
            longitude REAL,
            predicted_price REAL
        )
    ''')
    conn.commit()
    conn.close()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        # Get data from form
        try:
            features = [
                float(request.form['MedInc']),
                float(request.form['HouseAge']),
                float(request.form['AveRooms']),
                float(request.form['AveBedrms']),
                float(request.form['Population']),
                float(request.form['AveOccup']),
                float(request.form['Latitude']),
                float(request.form['Longitude'])
            ]
            
            # Predict
            final_features = [np.array(features)]
            prediction = model.predict(final_features)
            output = round(prediction[0], 2) # Price in $100,000s
            
            prediction_text = f"Estimated House Price: ${output * 100000:,.2f}"

            # Save to Database
            conn = sqlite3.connect('database.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (med_inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude, predicted_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (*features, output))
            conn.commit()
            conn.close()

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    init_db()  # Initialize DB on start
    app.run(debug=True)