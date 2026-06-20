import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import os

# Load and preprocess dataset
df = pd.read_csv("flight_dataset_final.csv")
df.columns = df.columns.str.strip()
df = df.fillna(df.median(numeric_only=True))

flight_dates = pd.to_datetime(dict(year=df['Year'], month=df['Month'], day=df['DayofMonth']))
df['DayofWeek'] = flight_dates.dt.dayofweek
df['CRSDepHour'] = df['CRSDepTime'].astype(str).str.zfill(4).str[:2].astype(int)
df['Delayed'] = (df['DepDelay'] > 15).astype(int)

le = LabelEncoder()
df['OriginCode'] = le.fit_transform(df['OriginAirportID'].astype(str))
df['DestCode'] = le.fit_transform(df['DestAirportID'].astype(str))

features = ['CRSDepHour', 'DayofWeek', 'Month', 'OriginCode', 'DestCode', 'temp', 'prcp', 'wspd']
X = df[features]
y = df['Delayed']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save scaler
os.makedirs("models", exist_ok=True)
joblib.dump(scaler, "models/scaler.pkl")

# Train and save traditional ML models
lr = LogisticRegression()
lr.fit(X_train, y_train)
joblib.dump(lr, "models/lr_model.pkl")

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
joblib.dump(dt, "models/dt_model.pkl")

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
joblib.dump(rf, "models/rf_model.pkl")

# Train and save LSTM model
X_lstm = X_scaled.reshape(-1, 1, X_scaled.shape[1])
X_train_lstm = X_lstm[:len(X_train)]
y_train_lstm = y_train.values

lstm_model = Sequential()
lstm_model.add(LSTM(32, input_shape=(1, X_scaled.shape[1])))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=10, verbose=1)
lstm_model.save("models/lstm_model.keras")

print("✅ All models trained and saved successfully.")