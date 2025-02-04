import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
# import gpustat
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score

# Configuration
config = {
    "input_dim": 768,  # Keep this if using embeddings like BERT; adjust if using different input features.
    "hidden_dim": 128,  # Increase to capture more complex patterns; try values like 128, 256, or 512.
    "output_dim": None,  # Set dynamically based on dataset.
    "dropout": 0.8,  # Increase dropout to prevent overfitting, especially with larger hidden dimensions.
    "learning_rate": 1e-3,  # Consider starting with a higher learning rate and using a learning rate scheduler.
    "batch_size": 128,  # Smaller batch sizes can help with generalization; adjust based on memory constraints.
    "num_epochs": 100,  # Increase the number of epochs to allow more training time; monitor for overfitting.
    "early_stopping_patience": 10  # This is reasonable; monitor validation performance to stop training early.
}

device = torch.device("cpu")
print("Device:", device)

# Load resampled data
X_resampled = np.load("smote/X_resampled_using_smote_lemmatized.npy")
Y_resampled = np.load("smote/Y_resampled_using_smote_lemmatized.npy")

# Convert numpy arrays to PyTorch tensors
X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32).to(device)
Y_resampled_tensor = torch.tensor(Y_resampled, dtype=torch.long).to(device)

# Print dataset shapes
print(f"X_resampled_tensor shape: {X_resampled_tensor.shape}")
print(f"Y_resampled_tensor shape: {Y_resampled_tensor.shape}")

# Split dataset into training, validation, and test sets
def split_dataset(X, Y, train_split=0.8, val_split=0.1):
    dataset_size = len(Y)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size
    indices = torch.randperm(dataset_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_data = TensorDataset(X[train_indices], Y[train_indices])
    val_data = TensorDataset(X[val_indices], Y[val_indices])
    test_data = TensorDataset(X[test_indices], Y[test_indices])

    return train_data, val_data, test_data

train_data, val_data, test_data = split_dataset(X_resampled_tensor, Y_resampled_tensor)

train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False)
test_loader = DataLoader(test_data, batch_size=config["batch_size"], shuffle=False)

print(f"Dataset size: {len(Y_resampled_tensor)}")
print(f"Training samples: {len(train_loader.dataset)}")
print(f"Validation samples: {len(val_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")
# Define the TransforMoji Model
# Define the DeepMoji-like Model
import torch
import torch.nn as nn

class BiLSTM(nn.Module): 
    def __init__(self, input_dim, output_dim, hidden_dim=128, lstm_layers=1, dropout=0.5):
        super(BiLSTM, self).__init__()
        
        # Bi-Directional LSTM
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers=lstm_layers, 
            bidirectional=True,  # Bi-directional for context capture
            batch_first=True, 
            dropout=dropout
        )
        
        # Fully connected layer for classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Multiply hidden_dim by 2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # lstm_out shape: (batch_size, seq_len, hidden_dim * 2)
        
        # Use the last hidden state for classification (e.g., pooling)
        # Summarize all time steps into a single vector (mean-pooling)
        pooled_out = torch.mean(lstm_out, dim=1)  # Shape: (batch_size, hidden_dim * 2)
        
        # Classification
        output = self.classifier(pooled_out)
        return output

# Load labels
data = pd.read_csv("final_data/final_lemmatized_dataset.csv")
Y = data['emojis'].values
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Initialize model
num_classes = len(label_encoder.classes_)
config["output_dim"] = num_classes
# Initialize the model
model = BiLSTM(
    input_dim=config["input_dim"], 
    output_dim=config["output_dim"], 
    hidden_dim=config["hidden_dim"], 
    lstm_layers=2,  # Two LSTM layers
    dropout=config["dropout"]  # Configured dropout
).to(device)

# Define function to evaluate model checkpoints
def evaluate_checkpoint(model_path, model, data_loader, label_encoder, dataset_name):
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc=f"Evaluating {model_path}"):
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{dataset_name} Evaluation for {model_path}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    return {
        "Checkpoint": model_path,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

# Paths to checkpoints
checkpoint_paths = [
    "trained_models/bilstm2_again_model_epoch_10.pth",
    "trained_models/bilstm2_again_model_epoch_50.pth",
    "trained_models/bilstm2_again_model_epoch_100.pth"
]


# Evaluate each checkpoint and collect results
results = []
for checkpoint in checkpoint_paths:
    result = evaluate_checkpoint(checkpoint, model, test_loader, label_encoder, "Test")
    results.append(result)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv("evaluation_metrics_bilstm.csv", index=False)
print("\nEvaluation results saved to evaluation_metrics_bilstm.csv")
