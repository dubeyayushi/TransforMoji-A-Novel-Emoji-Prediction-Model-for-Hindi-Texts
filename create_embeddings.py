import numpy as np
import pandas as pd
import emoji
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from numpy.linalg import norm
from tqdm import tqdm  # Import tqdm for the progress bar

# Check if MPS is available, else use CPU
device = torch.device("mps" if torch.backends.mps.is_built() else "cpu") 
print(device)

tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
model = AutoModel.from_pretrained("google/muril-base-cased")

# Move model to GPU
model = model.to(device)

data = pd.read_csv("final_data/final_lemmatized_dataset.csv")

X = data['lemmatized_content'].values

# Generate embeddings for all sentences
embeddings_list = []

# Wrap the iterable `X` with tqdm
for text in tqdm(X, desc="Processing sentences", unit="sentence"):
    # Tokenize the text and move tensors to GPU
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move inputs to GPU
    
    # Get embeddings
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # Embedding for each token
    
    # Sentence-level embedding (mean pooling)
    sentence_embedding = torch.mean(embeddings, dim=1).detach().cpu().numpy()  # Move back to CPU for numpy
    embeddings_list.append(sentence_embedding)

# Convert to numpy array
X_embeddings = np.vstack(embeddings_list)

np.save('embeddings/X_embeddings_lemmatized.npy', X_embeddings)