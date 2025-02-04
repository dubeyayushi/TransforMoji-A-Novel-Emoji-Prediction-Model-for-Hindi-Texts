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
    "dropout": 0.5,  # Increase dropout to prevent overfitting, especially with larger hidden dimensions.
    "learning_rate": 1e-3,  # Consider starting with a higher learning rate and using a learning rate scheduler.
    "batch_size": 128,  # Smaller batch sizes can help with generalization; adjust based on memory constraints.
    "num_epochs": 100,  # Increase the number of epochs to allow more training time; monitor for overfitting.
    "early_stopping_patience": 10  # This is reasonable; monitor validation performance to stop training early.
}


# # Check GPU status
# def check_gpu_status():
#     stats = gpustat.GPUStatCollection.new_query()
#     for gpu in stats.gpus:
#         print(gpu)

# check_gpu_status()

device = torch.device("cpu")
print("Device:", device)

# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
embedding_model = AutoModel.from_pretrained("google/muril-base-cased").to(device)

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

# Load emoji embeddings
emoji_embeddings = {}
with open("emoji2vec.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        emoji = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        emoji_embeddings[emoji] = vector

# Training Setup
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

def train_one_epoch(epoch, model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)
def cosine_similarity(vec1, vec2):
    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_model(loader, model, dataset_name):
    model.eval()
    total_similarity = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                pred_emoji = label_encoder.inverse_transform([pred])[0]
                true_emoji = label_encoder.inverse_transform([true])[0]

                if pred_emoji in emoji_embeddings and true_emoji in emoji_embeddings:
                    pred_vector = emoji_embeddings[pred_emoji]
                    true_vector = emoji_embeddings[true_emoji]
                    similarity = cosine_similarity(pred_vector, true_vector)
                    total_similarity += similarity
                    num_samples += 1

    avg_similarity = total_similarity / num_samples if num_samples > 0 else 0
    print(f"{dataset_name} Average Cosine Similarity: {avg_similarity:.4f}")
# Training Loop
for epoch in range(config["num_epochs"]):
    train_loss = train_one_epoch(epoch, model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")
    
    # Evaluate model
    evaluate_model(val_loader, model, "Validation")
    
    # Save model after each epoch
    torch.save(model.state_dict(), f"trained_models/bilstm2_again_model_epoch_{epoch + 1}.pth")
    print(f"Model saved after epoch {epoch + 1}")


# Optionally save the final model
torch.save(model.state_dict(), "trained_models/BiLSTM2_again_model_final.pth")
# Load the saved model
model.load_state_dict(torch.load("trained_models/BiLSTM2_again_model_final.pth"))
model.to(device)
print("Loaded the model from BiLSTM2_model_final.pth")

# Evaluate model on the test dataset
def evaluate_classification_metrics(loader, model, dataset_name):
    model.eval()
    y_true = []
    y_pred = []
    total_similarity = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc=f"Evaluating on {dataset_name}"):
            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                pred_emoji = label_encoder.inverse_transform([pred])[0]
                true_emoji = label_encoder.inverse_transform([true])[0]

                if pred_emoji in emoji_embeddings and true_emoji in emoji_embeddings:
                    pred_vector = emoji_embeddings[pred_emoji]
                    true_vector = emoji_embeddings[true_emoji]
                    similarity = cosine_similarity(pred_vector, true_vector)
                    total_similarity += similarity
                    num_samples += 1

    avg_similarity = total_similarity / num_samples if num_samples > 0 else 0
    print(f"{dataset_name} Average Cosine Similarity: {avg_similarity:.4f}")

    # Classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"{dataset_name} Classification Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Evaluate on the test dataset
# evaluate_classification_metrics(test_loader, model, "Test")


# Predict emoji for sentences
def predict_emoji_for_sentence(model, sentence):
    model.eval()
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        muril_output = embedding_model(**inputs)
        embeddings = muril_output.last_hidden_state
        embeddings_mean = embeddings.mean(dim=1).unsqueeze(0)
        outputs = model(embeddings_mean.to(device))
        _, predicted = torch.max(outputs, 1)

    return label_encoder.inverse_transform(predicted.cpu().numpy().reshape(-1))[0]

hindi_sentences = [
    "मुझे तुम्हारे साथ समय बिताना बहुत अच्छा लगा।",
    "यह जीत मेरे लिए बहुत खास है।",
    "तुम्हारी मुस्कान दिल को छू जाती है।",
    "आज मौसम बहुत सुहाना है।",
    "तुम्हारे उपहार ने मुझे बहुत खुश किया।",
    "मुझे यह बात सुनकर बहुत दुख हुआ।",
"आज का दिन मेरे लिए बहुत थकान भरा था।",
"मुझे इस समस्या का हल नहीं मिल रहा है।",
"यह सच में निराशाजनक है।",
"तुमने मेरी उम्मीदों पर पानी फेर दिया।",
"वह जगह इतनी खूबसूरत थी कि मैं देखता ही रह गया।",
"क्या यह सच में हो सकता है?",
"मुझे इस परिणाम की बिल्कुल भी उम्मीद नहीं थी।",
"तुमने यह कैसे किया? मैं हैरान हूं!",
"इतनी बड़ी खबर! मैं अपनी खुशी रोक नहीं पा रहा।",
"तुम्हारे साथ हर पल खास होता है।",
"तुम्हारी आवाज सुनकर मन को सुकून मिलता है।",
"तुम्हारे बिना जीवन अधूरा लगता है।",
"मुझे अंधेरे में बहुत डर लगता है।",
"क्या होगा अगर चीजें और खराब हो जाएं?",
    
]

for sentence in hindi_sentences:
    predicted_emoji = predict_emoji_for_sentence(model, sentence)
    print(f"Sentence: {sentence} -> Predicted Emoji: {predicted_emoji}")