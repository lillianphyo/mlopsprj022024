import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load model
def load_and_predict(model_path, data, seq_length, target_scaler):
    # Load the model
    model = load_model(model_path)

    # Create sequences
    X, y = create_sequences(data, seq_length)

    # Reshape for LSTM model
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

    # Make predictions
    predictions = model.predict(X)

    # Inverse transform predictions (use only the target scaler)
    predictions = target_scaler.inverse_transform(predictions)

    return predictions

# Function to create sequences
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # Use all columns except target
        y.append(data[i+seq_length, -1])  # Use only the target column
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Example dataset (replace with your actual dataset)
    data_df = pd.read_csv('data/ygn.csv')

    # Assume that 'c_rice' is the target column
    target_column = 'c_rice'
    feature_columns = ['o_rice', 'h_rice', 'l_rice', 'c_rice']

    # Scale all features except the target separately
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit the scalers
    feature_data = feature_scaler.fit_transform(data_df[feature_columns])
    target_data = target_scaler.fit_transform(data_df[[target_column]])

    # Combine scaled features and target
    scaled_data = np.hstack((feature_data[:, :-1], target_data))

    # Sequence length
    seq_length = 12

    # Load model and make predictions
    predictions = load_and_predict('model/lstm_model.h5', scaled_data, seq_length, target_scaler)

    # Print predictions
    print(predictions)
