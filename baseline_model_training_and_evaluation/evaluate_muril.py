import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import pandas as pd
import numpy as np

# Device configuration
device = torch.device("cpu")
print("Device:", device)

# Define the model class
class MuRILClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MuRILClassifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    

# Load data
X_test = np.load("smote/X_resampled_using_smote_lemmatized.npy")
Y_test = np.load("smote/Y_resampled_using_smote_lemmatized.npy")

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.long).to(device)

test_data = TensorDataset(X_test_tensor, Y_test_tensor)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Load label encoder
data = pd.read_csv("final_data/final_lemmatized_dataset.csv")
Y = data['emojis'].values
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Evaluation function
def evaluate_model(model, data_loader, label_encoder):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", unit="batch"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # Convert predictions and labels back to original label encoding
            predicted_labels = label_encoder.inverse_transform(predicted.cpu().numpy())
            true_labels = label_encoder.inverse_transform(labels.cpu().numpy())
            
            all_preds.extend(predicted_labels)
            all_labels.extend(true_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

# Paths to trained models
model_paths = {
    # "LSTM_10_epochs": "trained_models/lstm_smote_lemmatized_10_epochs.pth",
    # "LSTM_50_epochs": "trained_models/lstm_smote_lemmatized_50_epochs.pth",
    # "LSTM_100_epochs": "trained_models/lstm_smote_lemmatized_100_epochs.pth",
    # "RNN_10_epochs": "trained_models/rnn_smote_lemmatized_10_epochs.pth",
    # "RNN_50_epochs": "trained_models/rnn_smote_lemmatized_50_epochs.pth",
    # "RNN_100_epochs": "trained_models/rnn_smote_lemmatized_100_epochs.pth",
    # "BERT": "trained_models/bert_smote_lemmatized_test.pth",
    "MuRIL_10_epochs": "trained_models/muril_epoch_10.pth",
    "MuRIL_50_epochs": "trained_models/muril_epoch_50.pth",
    "MuRIL_100_epochs": "trained_models/muril_epoch_100.pth"
}

# Evaluate each model
results = {}
input_dim = X_test_tensor.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)

for model_name, model_path in model_paths.items():
    print(f"Loading and evaluating model: {model_name}")

    # Initialize the model and load weights
    model = MuRILClassifier(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Evaluate
    metrics = evaluate_model(model, test_loader, label_encoder)
    results[model_name] = metrics

    # Print metrics
    print(f"Metrics for {model_name}: {metrics}")

# Save results to a file
output_file = "evaluation_metrics_muril.json"
import json

with open(output_file, "w") as f:
    json.dump(results, f, indent=4)

print(f"Evaluation metrics saved to {output_file}")
