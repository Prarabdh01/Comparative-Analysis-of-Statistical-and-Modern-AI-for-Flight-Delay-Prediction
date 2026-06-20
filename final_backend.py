from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load models
scaler = joblib.load("models/scaler.pkl")
lr = joblib.load("models/lr_model.pkl")
dt = joblib.load("models/dt_model.pkl")
rf = joblib.load("models/rf_model.pkl")
lstm_model = load_model("models/lstm_model.keras")

@app.route("/")
def home():
    return render_template("index2.html")  # serve the HTML on /

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not (-50 <= data['temp'] <= 60):
            return jsonify({"error": "Temperature should be between -50 and 60 °C."})
        if data['prcp'] < 0:
            return jsonify({"error": "Precipitation cannot be negative."})
        if data['wspd'] > 150 or data['wspd'] < 0:
            return jsonify({"error": "Wind speed must be between 0 and 150 km/h."})
        
        input_features = np.array([
            data['CRSDepHour'],
            data['DayofWeek'],
            data['Month'],
            data['OriginCode'],
            data['DestCode'],
            data['temp'],
            data['prcp'],
            data['wspd']
        ]).reshape(1, -1)

        input_scaled = scaler.transform(input_features)
        input_lstm = input_scaled.reshape((1, 1, input_scaled.shape[1]))

        return jsonify({
            "Logistic Regression": int(lr.predict(input_scaled)[0]),
            "Decision Tree": int(dt.predict(input_scaled)[0]),
            "Random Forest": int(rf.predict(input_scaled)[0]),
            "LSTM": int((lstm_model.predict(input_lstm)[0][0] > 0.5))
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
