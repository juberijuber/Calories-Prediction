from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor

app = Flask(__name__)

# Load the trained model
model = XGBRegressor()
model.load_model("calorie_model.json")  # Save and load the trained model

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        gender = int(request.form['gender'])
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        body_temp = float(request.form['body_temp'])
        
        input_data = np.array([[gender, age, height, weight, duration, heart_rate, body_temp]])
        prediction = model.predict(input_data)[0]
        
        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)

