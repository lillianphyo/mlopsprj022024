# Within app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
# from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

# Load the model and the scaler
model_path = "model/lstm_model.h5"
scaler = MinMaxScaler(feature_range=(0, 1))

if os.path.exists(model_path):
    # model = load_model(model_path)
    model = tf.keras.models.load_model(model_path)
    print(f"Loaded model from {model_path}")
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
    return np.array(X)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON request data
        json_data = request.get_json()

        # Convert to DataFrame
        data_df = pd.DataFrame(json_data)

        # Assume that 'c_rice' is the target column
        target_column = 'c_rice'
        feature_columns = ['o_rice', 'h_rice', 'l_rice', 'c_rice']

        # Scale all features except the target separately
        feature_scaler = MinMaxScaler(feature_range=(0, 1))
        target_scaler = MinMaxScaler(feature_range=(0, 1))

        # Scale the features and the target separately
        feature_data = feature_scaler.transform(data_df[feature_columns])
        target_data = target_scaler.transform(data_df[[target_column]])

        # Combine scaled features and target
        scaled_data = np.hstack((feature_data[:, :-1], target_data))

        # Define the sequence length (same as used in training)
        seq_length = 12

        # Create sequences
        X_new = create_sequences(scaled_data, seq_length)

        # Reshape X_new for the LSTM model
        X_new = X_new.reshape((X_new.shape[0], X_new.shape[1], X_new.shape[2]))

        # Make predictions
        predictions = model.predict(X_new)

        # Inverse transform to get the original scale (only for the target)
        predictions = target_scaler.inverse_transform(predictions)

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)