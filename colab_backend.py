# Install necessary packages
!pip install pyngrok flask pandas scikit-learn tensorflow matplotlib seaborn flask-cors

# Import everything
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ----------------------- Flask app -----------------------
app = Flask(__name__)
CORS(app)  # optional, if your frontend runs on another port or machine

# ----------------------- Load and preprocess data -----------------------
df = pd.read_csv("flight_dataset_final.csv")
df.columns = df.columns.str.strip()
df = df.fillna(df.median(numeric_only=True))

# Extract DayOfWeek and CRSDepHour
flight_dates = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=df['DayofMonth']))
df['DayofWeek'] = flight_dates.dt.dayofweek
df['CRSDepHour'] = df['CRSDepTime'].astype(str).str.zfill(4).str[:2].astype(int)

# Binary target variable
df['Delayed'] = (df['DepDelay'] > 15).astype(int)

# Encode categorical variables
le = LabelEncoder()
df['OriginCode'] = le.fit_transform(df['OriginAirportID'].astype(str))
df['DestCode'] = le.fit_transform(df['DestAirportID'].astype(str))

features = ['CRSDepHour', 'DayofWeek', 'Month', 'OriginCode', 'DestCode', 'temp', 'prcp', 'wspd']
X = df[features]
y = df['Delayed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# ----------------------- Train models -----------------------

lr = LogisticRegression()
lr.fit(X_train, y_train)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Prepare data for LSTM
X_lstm = X_scaled.reshape(-1, 1, X_scaled.shape[1])
X_train_lstm = X_lstm[:len(X_train)]
X_test_lstm = X_lstm[len(X_train):]
y_train_lstm = y_train.values
y_test_lstm = y_test.values

model = Sequential()
model.add(LSTM(32, input_shape=(1, X_scaled.shape[1])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=10, verbose=0)

# ----------------------- Evaluation function -----------------------

def evaluate_all_metrics(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    cmap_dict = {
        "Logistic Regression": "Blues",
        "Decision Tree": "Greens",
        "Random Forest": "Oranges",
        "LSTM": "Purples"
    }

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_dict.get(model_name, "Blues"))
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    print(f"📊 {model_name} Metrics:")
    print(f"  - Accuracy:  {accuracy:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1 Score:  {f1:.4f}")
    print("-" * 40)

# Evaluate all models
evaluate_all_metrics("Logistic Regression", y_test, lr.predict(X_test))
evaluate_all_metrics("Decision Tree", y_test, dt.predict(X_test))
evaluate_all_metrics("Random Forest", y_test, rf.predict(X_test))
lstm_preds = (model.predict(X_test_lstm) > 0.5).astype(int)
evaluate_all_metrics("LSTM", y_test_lstm, lstm_preds)

# ----------------------- Flask /predict endpoint -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        # Basic validation
        if not (-50 <= data['temp'] <= 60):
            return jsonify({"error": "Temperature should be between -50 and 60 °C."})
        if data['prcp'] < 0:
            return jsonify({"error": "Precipitation cannot be negative."})
        if not (0 <= data['wspd'] <= 150):
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

        lr_pred = int(lr.predict(input_scaled)[0])
        dt_pred = int(dt.predict(input_scaled)[0])
        rf_pred = int(rf.predict(input_scaled)[0])
        lstm_pred = int((model.predict(input_lstm)[0][0] > 0.5))

        return jsonify({
            "Logistic Regression": lr_pred,
            "Decision Tree": dt_pred,
            "Random Forest": rf_pred,
            "LSTM": lstm_pred
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# ----------------------- Run Flask app -----------------------
if __name__ == "__main__":
    # Set your ngrok authtoken here (replace YOUR_AUTHTOKEN)
    !ngrok authtoken 2xe1YtskYLFaj1aXVoKi484B5kE_6e8zBeX5Awdbw27Q4sMwX
    
    public_url = ngrok.connect(5000)
    print(f" * ngrok tunnel URL: {public_url}")
    app.run(port=5000)
