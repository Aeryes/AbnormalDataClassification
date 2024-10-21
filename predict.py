import pandas as pd
import numpy as np
import joblib
import torch
from model import SimpleDNN, Autoencoder, LSTMNetwork  # Import the models from model.py


def load_model(model_path, model_type='sklearn', input_size=None):
    """Load the trained model based on its type."""
    if model_type == 'sklearn':
        model = joblib.load(model_path)
    elif model_type == 'pytorch':
        # Load PyTorch model and initialize the appropriate model class
        if 'dnn' in model_path.lower():
            model = SimpleDNN(input_size)
        elif 'autoencoder' in model_path.lower():
            model = Autoencoder(input_size)
        elif 'lstm' in model_path.lower():
            model = LSTMNetwork(input_size)
        else:
            raise ValueError("Unsupported PyTorch model type")
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set to evaluation mode for inference
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    print(f"Model loaded from {model_path}")
    return model


def load_data(file_path='data/data_predict.csv'):
    """Load data for prediction."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Feature selection and preprocessing."""
    features = ['Source Port', 'Destination Port', 'Protocol', 'Length']
    X = data[features]
    X['Protocol'] = X['Protocol'].astype('category').cat.codes
    return X


def make_predictions(model, X, model_name, model_type='sklearn'):
    """Make predictions using the provided model."""
    if model_type == 'sklearn' or model_type == 'pyod':
        # Predict using sklearn or PyOD models
        predictions = model.predict(X)

        if model_name in ['IsolationForest', 'OneClassSVM']:
            predictions = np.where(predictions == -1, 1, 0)  # Convert -1 (anomaly) to 1 and 1 (normal) to 0
        elif model_name == 'RandomForest':
            # Swap 1 and 0 to correctly reflect anomalies (0 for anomalies, 1 for normal)
            predictions = np.where(predictions == 1, 0, 1)

    elif model_type == 'pytorch':
        # Convert input data to PyTorch tensor for prediction
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        if 'lstm' in model_name.lower():
            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension for LSTM
        with torch.no_grad():
            outputs = model(X_tensor)
            # Convert outputs to binary predictions (0 or 1)
            predictions = (outputs >= 0.5).float().numpy().flatten()
    else:
        raise ValueError(f"The loaded model '{model_name}' does not support the 'predict' method.")
    return predictions



if __name__ == "__main__":
    # Define the models and their paths, including both sklearn and PyTorch models
    model_paths = {
        'RandomForest': ('random_forest_model.pkl', 'sklearn'),
        'HistGradientBoosting': ('hist_gradient_boosting_model.pkl', 'sklearn'),
        'IsolationForest': ('isolation_forest_model.pkl', 'sklearn'),
        'SimpleDNN': ('simplednn.pth', 'pytorch'),
        'Autoencoder': ('autoencoder.pth', 'pytorch'),
        'LSTM': ('lstm.pth', 'pytorch')
    }

    # Load and preprocess the data
    data = load_data('data/data.csv')  # Specify the correct path to your data
    X = preprocess_data(data)

    # Store predictions from each model
    predictions_dict = {}

    # Iterate through each model and make predictions
    input_size = X.shape[1]  # Used for PyTorch models
    for model_name, (model_path, model_type) in model_paths.items():
        try:
            # Load the model
            model = load_model(model_path, model_type=model_type, input_size=input_size)

            # Make predictions
            predictions = make_predictions(model, X, model_name, model_type=model_type)
            predictions_dict[model_name] = predictions

            # Output the number of anomalies detected
            num_anomalies = np.sum(predictions)
            print(f"\nModel: {model_name}")
            print(f"Predictions: {predictions}")
            print(f"Number of anomalies detected: {num_anomalies} out of {len(predictions)} samples")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")

    # Optionally: Compare predictions between models
    print("\nSummary of Anomalies Detected by Each Model:")
    for model_name, predictions in predictions_dict.items():
        print(f"{model_name}: {np.sum(predictions)} anomalies detected")
