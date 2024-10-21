import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import models from model.py
from model import SimpleDNN, Autoencoder, LSTMNetwork


def load_data(file_path='data/data.csv'):
    """Load data from a CSV file."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Feature selection, preprocessing, and imputation."""
    features = ['Source Port', 'Destination Port', 'Protocol', 'Length']
    X = data[features]

    # Convert 'Protocol' to numerical values (if it's categorical)
    X['Protocol'] = X['Protocol'].astype('category').cat.codes

    # Handle missing values using mean imputation
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=features)

    # Use 'bad_packet' as the label
    y = data['bad_packet']

    return X, y


def get_device():
    """Check if GPU is available and return the appropriate device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_random_forest(X_train, y_train, X_test, y_test):
    """Train a Random Forest classifier and evaluate it."""
    print("\nTraining RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluate the model
    y_pred_test = rf.predict(X_test)
    metrics = evaluate_model("RandomForest", y_test, y_pred_test)
    return rf, metrics


def train_hist_gradient_boosting(X_train, y_train, X_test, y_test):
    """Train a HistGradientBoosting classifier and evaluate it."""
    print("\nTraining HistGradientBoostingClassifier...")
    hgb = HistGradientBoostingClassifier(random_state=42)
    hgb.fit(X_train, y_train)

    # Evaluate the model
    y_pred_test = hgb.predict(X_test)
    metrics = evaluate_model("HistGradientBoosting", y_test, y_pred_test)
    return hgb, metrics


def train_isolation_forest(X_train, X_test):
    """Train an Isolation Forest model and evaluate it."""
    print("\nTraining IsolationForest...")
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_forest.fit(X_train)

    # Predict anomalies (-1 for outliers, 1 for inliers)
    y_pred_test = isolation_forest.predict(X_test)
    y_pred_test = np.where(y_pred_test == -1, 1, 0)

    metrics = evaluate_model("IsolationForest", y_pred_test, y_pred_test)
    return isolation_forest, metrics


def train_dnn(X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    """Train a deep neural network using PyTorch and evaluate it."""
    print("\nTraining SimpleDNN with PyTorch...")
    input_size = X_train.shape[1]
    model = SimpleDNN(input_size)

    # Use GPU if available
    device = get_device()
    model.to(device)

    # Convert data to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test = (y_pred_test >= 0.5).float().cpu().numpy().flatten()

    y_test_np = y_test.to_numpy()
    metrics = evaluate_model("SimpleDNN", y_test_np, y_pred_test)

    # Save the trained model
    torch.save(model.state_dict(), "simplednn.pth")
    print(f"Model saved to {"simplednn.pth"}")

    return model, metrics


def train_autoencoder(X_train, X_test, epochs=50, batch_size=64):
    """Train an autoencoder for anomaly detection using PyTorch and evaluate it."""
    print("\nTraining Autoencoder with PyTorch...")
    input_size = X_train.shape[1]
    model = Autoencoder(input_size)

    # Use GPU if available
    device = get_device()
    model.to(device)

    # Convert data to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

    # DataLoader
    train_loader = DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate the model using reconstruction error
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_test_tensor)
        reconstruction_error = nn.functional.mse_loss(reconstructed, X_test_tensor, reduction='none')
        error_per_sample = reconstruction_error.mean(dim=1).cpu().numpy()
        y_pred_test = (error_per_sample > np.percentile(error_per_sample, 95)).astype(int)

    y_test_np = y_test.to_numpy()
    metrics = evaluate_model("Autoencoder", y_test_np, y_pred_test)

    # Save the trained model
    torch.save(model.state_dict(), "autoencoder.pth")
    print(f"Model saved to {"autoencoder.pth"}")

    return model, metrics


def train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=64):
    """Train an LSTM-based network using PyTorch and evaluate it."""
    print("\nTraining LSTM with PyTorch...")
    input_size = X_train.shape[1]
    model = LSTMNetwork(input_size)

    # Use GPU if available
    device = get_device()
    model.to(device)

    # Convert data to PyTorch tensors and move them to the device
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor)
        y_pred_test = (y_pred_test >= 0.5).float().cpu().numpy().flatten()

    y_test_np = y_test.to_numpy()
    metrics = evaluate_model("LSTM", y_test_np, y_pred_test)

    # Save the trained model
    torch.save(model.state_dict(), "lstm.pth")
    print(f"Model saved to {"lstm.pth"}")

    return model, metrics


def evaluate_model(model_name, y_true, y_pred):
    """Evaluate model performance using accuracy, precision, recall, and F1-score."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Print the metrics
    print(f"\n{model_name} Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    # Return metrics as a dictionary
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }


def plot_metrics(metrics_list):
    """Plot the performance metrics of various models for comparison."""
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df.set_index('Model').plot(kind='bar', figsize=(10, 6))
    plt.title('Model Performance Metrics')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Load and preprocess data
    data = load_data()
    X, y = preprocess_data(data)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Store metrics for each model
    metrics_list = []

    # Train and evaluate RandomForestClassifier
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_test, y_test)
    metrics_list.append(rf_metrics)

    # Train and evaluate HistGradientBoostingClassifier
    hgb_model, hgb_metrics = train_hist_gradient_boosting(X_train, y_train, X_test, y_test)
    metrics_list.append(hgb_metrics)

    # Train and evaluate IsolationForest
    isolation_forest_model, isolation_forest_metrics = train_isolation_forest(X_train, X_test)
    metrics_list.append(isolation_forest_metrics)

    # Train and evaluate the DNN with PyTorch
    dnn_model, dnn_metrics = train_dnn(X_train, y_train, X_test, y_test, epochs=50, batch_size=64)
    metrics_list.append(dnn_metrics)

    # Train and evaluate the Autoencoder with PyTorch
    autoencoder_model, autoencoder_metrics = train_autoencoder(X_train, X_test)
    metrics_list.append(autoencoder_metrics)

    # Train and evaluate the LSTM with PyTorch
    lstm_model, lstm_metrics = train_lstm(X_train, y_train, X_test, y_test, epochs=50, batch_size=64)
    metrics_list.append(lstm_metrics)

    # Plot the comparison of the end-of-training metrics
    plot_metrics(metrics_list)
