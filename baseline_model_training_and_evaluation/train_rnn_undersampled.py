import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from numpy.linalg import norm
from transformers import AutoTokenizer, AutoModel
from imblearn.under_sampling import RandomUnderSampler

device = torch.device("cpu")
print("Device: ", device)

# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
embedding_model = AutoModel.from_pretrained("google/muril-base-cased").to(device)

data = pd.read_csv("final_data/final_lemmatized_dataset.csv")

# Separate features and labels
X = np.load("embeddings/X_embeddings_lemmatized.npy")  # Assuming embeddings are stored here
Y = data["emojis"].values

# Encode labels
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

# Apply undersampling
undersampler = RandomUnderSampler(random_state=42)
X_resampled, Y_resampled = undersampler.fit_resample(X, Y_encoded)

print(f"Original dataset size: {len(Y)}")
print(f"Resampled dataset size: {len(Y_resampled)}")

# Convert the numpy arrays to PyTorch tensors and move to the desired device (GPU or CPU)
X_resampled_tensor = torch.tensor(X_resampled, dtype=torch.float32).to("cpu")
Y_resampled_tensor = torch.tensor(Y_resampled, dtype=torch.long).to("cpu")

# Now you can use X_resampled_tensor and Y_resampled_tensor for training
print(f"X_resampled_tensor shape: {X_resampled_tensor.shape}")
print(f"Y_resampled_tensor shape: {Y_resampled_tensor.shape}")

dataset_size = len(Y_resampled_tensor)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

indices = torch.randperm(dataset_size)
train_indices, val_indices = indices[:train_size], indices[train_size:]

train_data = TensorDataset(X_resampled_tensor[train_indices], Y_resampled_tensor[train_indices])
val_data = TensorDataset(X_resampled_tensor[val_indices], Y_resampled_tensor[val_indices])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

total_samples_in_train = len(train_loader.dataset)
total_samples_in_val = len(val_loader.dataset)
print(f"Dataset size: {dataset_size}")
print(f"Total samples in training dataset: {total_samples_in_train}")
print(f"Total samples in validation dataset: {total_samples_in_val}")

class EmojifierRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(EmojifierRNN, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, nonlinearity='tanh')
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hidden = self.rnn(x)
        hidden = hidden[-1]  # Take the last hidden state
        out = self.fc(hidden)
        return out

# Model configuration
input_dim = X_resampled_tensor.shape[1]
hidden_dim = 128
output_dim = len(label_encoder.classes_)
learning_rate = 0.001
num_epochs = 100

model = EmojifierRNN(input_dim, hidden_dim, output_dim).to("cpu")
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", leave=True) as t:
            for inputs, labels in t:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)  # Add sequence length dimension
                optimizer.zero_grad()
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update weights
                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (t.n + 1))  # Update progress bar with average loss

print("Training the model...")
train_model(model, train_loader, criterion, optimizer, num_epochs)

save_path = "trained_models/rnn_undersampling_lemmatized_100_epochs.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

# Load emoji embeddings for evaluation
emoji_embeddings = {}
with open("emoji2vec.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        emoji = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        emoji_embeddings[emoji] = vector

# Cosine similarity function
def cosine_similarity(vec1, vec2):
    vec1 = np.squeeze(vec1)  # Remove single-dimensional entries from vec1
    vec2 = np.squeeze(vec2)  # Remove single-dimensional entries from vec2
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_model(model, val_loader, emoji_embeddings, label_encoder):
    model.eval()
    total_similarity = 0
    num_samples = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Evaluating", unit="batch") as t:
            for inputs, labels in t:
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
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

print("Evaluating the model...")
evaluate_model(model, val_loader, emoji_embeddings, label_encoder)

def predict_emoji_for_sentence(model, sentence, tokenizer, label_encoder, embedding_model):
    model.eval()  # Set the model to evaluation mode
    
    # Tokenize the input sentence and get embeddings using MuRIL
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    
    # Move to device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        # Generate embeddings using MuRIL
        muril_output = embedding_model(**inputs)
        embeddings = muril_output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Compute sentence embedding using mean pooling
        sentence_embedding = torch.mean(embeddings, dim=1)  # Shape: [batch_size, hidden_dim]
        
        # Pass the sentence embedding through the LSTM model
        outputs = model(sentence_embedding)  # LSTM expects embeddings as input (no need to add seq_dim here)
        
        # Ensure the output has the correct shape
        if len(outputs.shape) > 1:
            # If the output is 2D (batch_size, num_classes), use torch.max
            _, predicted = torch.max(outputs, 1) 
        else:
            # If the output is 1D (batch_size), no need for max, use the index directly
            predicted = outputs.argmax(dim=0)

    # Convert predicted to a 1D array if it's a scalar (batch size = 1)
    predicted_emoji = label_encoder.inverse_transform(predicted.cpu().numpy().reshape(-1))[0]
    return predicted_emoji

# Example Hindi sentences for prediction
hindi_sentences = [
    "आज का दिन बहुत अच्छा रहा।",  # Today was a very good day.
    "मुझे यह जगह बहुत पसंद आई।",  # I really liked this place.
    "तुम्हारा काम वाकई शानदार है।",  # Your work is truly amazing.
    "मुझे यह सुनकर बहुत दुख हुआ।",  # I was very sad to hear this.
    "मैं इस स्थिति से परेशान हूं।",  # I am upset about this situation.
    "मुझे इस बात पर बहुत गुस्सा आया।",  # I was very angry about this.
    "आज मौसम थोड़ा ठंडा है।",  # The weather is a bit cold today.
    "मुझे कल का प्रोजेक्ट तैयार करना है।",  # I need to prepare tomorrow's project.
    "क्या तुमने वह नई फिल्म देखी?",  # Did you watch that new movie?
    "मुझे नई नौकरी मिल गई!",  # I got a new job!
    "हमारी टीम ने मैच जीत लिया।",  # Our team won the match.
    "मैं अगले हफ्ते यात्रा पर जा रहा हूं।",  # I am going on a trip next week.
    "आज मैं थोड़ी उदास हूं।",  # I am feeling a bit sad today.
    "परीक्षा में अच्छे नंबर नहीं आए।",  # Didn't get good marks in the exam.
    "मुझे अपने दोस्त की बहुत याद आ रही है।",  # I am missing my friend a lot.
    "मुझे अंधेरे से डर लगता है।",  # I am afraid of the dark.
    "क्या यह काम समय पर पूरा होगा?",  # Will this work be completed on time?
    "मुझे परीक्षा का बहुत तनाव है।",  # I am very stressed about the exam.
    "तुम हमेशा देर से क्यों आते हो?",  # Why are you always late?
    "मुझे इस निर्णय से बिल्कुल सहमत नहीं हूं।"  # I completely disagree with this decision.
]

# Predict emojis for the sentences
for sentence in hindi_sentences:
    predicted_emoji = predict_emoji_for_sentence(model, sentence, tokenizer, label_encoder, embedding_model)
    print(f"Text: {sentence}\nPredicted Emoji: {predicted_emoji}\n")
