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

device = torch.device("mps" if torch.backends.mps.is_built() else "cpu")  # Use GPU if available
print("Device: ", device)

# Load embeddings and labels
X_embeddings = np.load("embeddings/X_embeddings_lemmatized.npy")
data = pd.read_csv("final_data/final_lemmatized_dataset.csv")
Y = data['emojis'].values

# Load the embedding model
tokenizer = AutoTokenizer.from_pretrained("google/muril-base-cased")
embedding_model = AutoModel.from_pretrained("google/muril-base-cased").to(device)

# Encode emoji labels into numeric format
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)

smote = SMOTE(random_state=42)

# Wrap the SMOTE resampling in tqdm to track progress
print("Applying SMOTE for class balancing...")

# Create a tqdm instance for progress tracking
with tqdm(total=X_embeddings.shape[0], desc="SMOTE Resampling", unit="sample") as pbar:
    # Use tqdm to track progress
    X_resampled, Y_resampled = smote.fit_resample(X_embeddings, Y_encoded)

    # Update progress bar once the resampling is done
    pbar.update(X_embeddings.shape[0])

# Save resampled data for future use
np.save("smote/X_resampled_using_smote_lemmatized.npy", X_resampled)
np.save("smote/Y_resampled_using_smote_lemmatized.npy", Y_resampled)

print("SMOTE complete!")