import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cpu")
print("Device: ", device)

# Load MuRIL tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
muril_model = AutoModel.from_pretrained("google/muril-base-cased")
muril_model = muril_model.to(device)

# Load the dataset
data = pd.read_csv("final_data/final_lemmatized_dataset.csv")
Y = data['emojis'].values
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

X_resampled = np.load("smote/X_resampled_using_smote_lemmatized.npy")
Y_resampled = np.load("smote/Y_resampled_using_smote_lemmatized.npy")

X_tensor = torch.tensor(X_resampled, dtype=torch.float32).to(device)
Y_tensor = torch.tensor(Y_resampled, dtype=torch.long).to(device)

# Split into training and validation sets
dataset_size = len(Y_tensor)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

indices = torch.randperm(dataset_size)
train_indices, val_indices = indices[:train_size], indices[train_size:]

train_data = TensorDataset(X_tensor[train_indices], Y_tensor[train_indices])
val_data = TensorDataset(X_tensor[val_indices], Y_tensor[val_indices])

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

# Define the classification head for MuRIL embeddings
class MuRILClassifier(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MuRILClassifier, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)
    
input_dim = X_resampled.shape[1]
output_dim = len(np.unique(Y_resampled))
model = MuRILClassifier(input_dim, output_dim).to(device)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training function
# def train_model(model, train_loader, criterion, optimizer, num_epochs):
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as t:
#             for inputs, labels in t:
#                 optimizer.zero_grad()
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 t.set_postfix(loss=running_loss / (t.n + 1))
        
#         # Save checkpoint every 10 epochs
#         if (epoch + 1) % 10 == 0:
#             save_path = f"trained_models/muril_epoch_{epoch + 1}.pth"
#             torch.save(model.state_dict(), save_path)
#             print(f"Checkpoint saved to {save_path}")

# print("Training the MuRIL model...")
# train_model(model, train_loader, criterion, optimizer, num_epochs=100)

# # Save the final model
# final_save_path = "trained_models/muril_100_epochs.pth"
# torch.save(model.state_dict(), final_save_path)
# print(f"Model saved to {final_save_path}")


# Load the checkpoint
checkpoint_path = "trained_models/muril_epoch_90.pth"
model.load_state_dict(torch.load(checkpoint_path))
print(f"Model loaded from {checkpoint_path}")

# Define a new training function to resume
def resume_training(model, train_loader, criterion, optimizer, start_epoch, total_epochs):
    model.train()
    for epoch in range(start_epoch, total_epochs):
        running_loss = 0.0
        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}", unit="batch") as t:
            for inputs, labels in t:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                t.set_postfix(loss=running_loss / (t.n + 1))
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = f"trained_models/muril_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    # Save the final model
    final_save_path = f"trained_models/muril_epoch_{total_epochs}.pth"
    torch.save(model.state_dict(), final_save_path)
    print(f"Model saved to {final_save_path}")

# Resume training from epoch 90
start_epoch = 90
total_epochs = 100
resume_training(model, train_loader, criterion, optimizer, start_epoch, total_epochs)

emoji_embeddings = {}
with open(r"emoji2vec.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split()
        emoji = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        emoji_embeddings[emoji] = vector

# Evaluation function
def evaluate_model(model, val_loader, emoji_embeddings, label_encoder):
    model.eval()
    total_similarity = 0
    num_samples = 0

    with torch.no_grad():
        with tqdm(val_loader, desc="Evaluating", unit="batch") as t:
            for inputs, labels in t:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)

                for pred, true in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    pred_emoji = label_encoder.inverse_transform([pred])[0]
                    true_emoji = label_encoder.inverse_transform([true])[0]

                    if pred_emoji in emoji_embeddings and true_emoji in emoji_embeddings:
                        pred_vector = emoji_embeddings[pred_emoji]
                        true_vector = emoji_embeddings[true_emoji]
                        similarity = cosine_similarity([pred_vector], [true_vector])[0][0]
                        total_similarity += similarity
                        num_samples += 1

    avg_similarity = total_similarity / num_samples if num_samples > 0 else 0
    print(f"Average Cosine Similarity: {avg_similarity:.4f}")

print("Evaluating the model...")
evaluate_model(model, val_loader, emoji_embeddings, label_encoder)

# Prediction function
def predict_lines(model, lines, tokenizer, label_encoder):
    model.eval()
    for line in lines:
        inputs = tokenizer(line, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            muril_output = muril_model(**inputs.to(device))
            embeddings = muril_output.last_hidden_state
            sentence_embedding = torch.mean(embeddings, dim=1)
            outputs = model(sentence_embedding)
            _, predicted = torch.max(outputs, 1)

        predicted_emoji = label_encoder.inverse_transform(predicted.cpu().numpy())[0]
        print(f"Text: {line}\nPredicted Emoji: {predicted_emoji}\n")

hindi_lines = [
    "आज का दिन बहुत अच्छा रहा।",
    "मुझे यह जगह बहुत पसंद आई।",
    "तुम्हारा काम वाकई शानदार है।",
    "मुझे यह सुनकर बहुत दुख हुआ।",
    "मैं इस स्थिति से परेशान हूं।",
    "मुझे इस बात पर बहुत गुस्सा आया।",
    "आज मौसम थोड़ा ठंडा है।",
    "मुझे कल का प्रोजेक्ट तैयार करना है।",
    "क्या तुमने वह नई फिल्म देखी?",
    "मुझे नई नौकरी मिल गई!",
    "हमारी टीम ने मैच जीत लिया।",
    "मैं अगले हफ्ते यात्रा पर जा रहा हूं।",
    "आज मैं थोड़ी उदास हूं।",
    "परीक्षा में अच्छे नंबर नहीं आए।",
    "मुझे अपने दोस्त की बहुत याद आ रही है।",
    "मुझे अंधेरे से डर लगता है।",
    "क्या यह काम समय पर पूरा होगा?",
    "मुझे परीक्षा का बहुत तनाव है।",
    "तुम हमेशा देर से क्यों आते हो?",
    "मुझे इस निर्णय से बिल्कुल सहमत नहीं हूं।"
]

print("Predictions for sample Hindi lines...")
predict_lines(model, hindi_lines, tokenizer, label_encoder)

