import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import torch
from model import SimpleDNN, Autoencoder, LSTMNetwork  # Import models from model.py


def load_data(file_path='data/data.csv'):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Feature selection and preprocessing."""
    features = ['Source Port', 'Destination Port', 'Protocol', 'Length']
    X = data[features]

    # Convert 'Protocol' to numerical values (if it's categorical)
    X['Protocol'] = X['Protocol'].astype('category').cat.codes

    # Use 'bad_packet' as the label
    y = data['bad_packet']

    return X, y


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


def make_predictions(model, X, model_name, model_type='sklearn'):
    """Make predictions using the provided model."""
    if model_type == 'sklearn' or model_type == 'pyod':
        # Predict using sklearn or PyOD models
        predictions = model.predict(X)
        if model_name in ['IsolationForest', 'OneClassSVM']:
            predictions = np.where(predictions == -1, 1, 0)  # Convert -1 (anomaly) to 1 and 1 (normal) to 0
    elif model_type == 'pytorch':
        # Convert input data to PyTorch tensor for prediction
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        if 'lstm' in model_name.lower():
            X_tensor = X_tensor.unsqueeze(1)  # Add sequence dimension for LSTM
        with torch.no_grad():
            outputs = model(X_tensor)
            predictions = (outputs >= 0.5).float().numpy().flatten()  # Convert to binary predictions
    else:
        raise ValueError(f"The loaded model '{model_name}' does not support the 'predict' method.")
    return predictions


def evaluate_model(y_true, y_pred, model_name):
    """Evaluate the model using various metrics and display results."""
    # Show detailed classification report
    report = classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly'], output_dict=True)
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))

    # Confusion matrix for the test set
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Normal', 'Anomaly'])
    disp.plot()
    plt.title(f'Confusion Matrix: {model_name}')
    plt.show()

    return report, conf_matrix


def compare_models(models, model_paths, X, y_true):
    """Compare multiple models and display their evaluation results."""
    reports = {}
    confusion_matrices = {}

    for model_name, (model_path, model_type) in model_paths.items():
        try:
            input_size = X.shape[1]  # Number of features for PyTorch models
            # Load the model
            model = load_model(model_path, model_type=model_type, input_size=input_size)

            # Make predictions
            y_pred = make_predictions(model, X, model_name, model_type=model_type)

            # Evaluate the model and store the report and confusion matrix
            report, conf_matrix = evaluate_model(y_true, y_pred, model_name)
            reports[model_name] = report
            confusion_matrices[model_name] = conf_matrix
        except Exception as e:
            print(f"Error with model {model_name}: {e}")

    # Create a summary dataframe for comparison
    summary_df = pd.DataFrame({
        model_name: {
            "Precision (Anomaly)": report["Anomaly"]["precision"],
            "Recall (Anomaly)": report["Anomaly"]["recall"],
            "F1-Score (Anomaly)": report["Anomaly"]["f1-score"],
            "Precision (Normal)": report["Normal"]["precision"],
            "Recall (Normal)": report["Normal"]["recall"],
            "F1-Score (Normal)": report["Normal"]["f1-score"],
            "Accuracy": report["accuracy"]
        }
        for model_name, report in reports.items()
    }).T

    # Display the summary dataframe
    print("\nComparison Summary:")
    print(summary_df)

    # Visualize the summary for a better comparison
    summary_df.plot(kind='bar', figsize=(12, 8), title='Model Comparison Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return reports, confusion_matrices


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
    data = load_data()
    X, y_true = preprocess_data(data)

    # Compare all models and output the evaluation
    compare_models(models=model_paths.keys(), model_paths=model_paths, X=X, y_true=y_true)
