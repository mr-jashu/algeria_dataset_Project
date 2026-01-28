import pickle
from flask import Flask,jsonify,render_template,request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)

##import ridge regressor and standard scaler
ridge_model = pickle.load(open('Models/ridge.pkl', 'rb'))
standard_Scaler = pickle.load(open('Models/scaler.pkl', 'rb'))


@app.route("/predict_data", methods=["GET","POST"])
def predict():
    if request.method == "POST":
        try:
            # Get JSON data from request
            data = request.get_json()
            
            # Extract features in the correct order
            # Based on the dataset: Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, Classes
            features = [
                float(data.get('Temperature', 0)),
                float(data.get('RH', 0)),
                float(data.get('Ws', 0)),
                float(data.get('Rain', 0)),
                float(data.get('FFMC', 0)),
                float(data.get('DMC', 0)),
                float(data.get('DC', 0)),
                float(data.get('ISI', 0)),
                float(data.get('Classes', 0))
            ]
            
            # Scale the input data
            scaled_data = standard_Scaler.transform(np.array(features).reshape(1, -1))
            
            # Make prediction
            prediction = ridge_model.predict(scaled_data)
            
            return jsonify({
                'prediction': float(prediction[0]),
                'error': None
            })
        except Exception as e:
            return jsonify({
                'prediction': None,
                'error': str(e)
            }), 400
    else:
        return render_template("index.html")
        # return jsonify({
        #     'message': 'Send a POST request with input data to get predictions.'
        # }) , 450

@app.route("/")
def index():
    return render_template("home.html")   

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
    