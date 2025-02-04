import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

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

X_train, X_test, Y_train, Y_test = train_test_split(
    X_resampled, Y_resampled, test_size=0.2, random_state=42
)

# Model paths
model_paths = [
    'trained_models/svm_model_linear.joblib',
    'trained_models/svm_model_poly.joblib',
    'trained_models/svm_model_rbf.joblib',
    'trained_models/svm_model_sigmoid.joblib'
]

# Evaluation results
results = {}

for model_path in model_paths:
    # Load model
    print(f"Evaluating {model_path}")
    model = joblib.load(model_path)
    
    # Predict
    Y_pred = model.predict(X_test)
    
    # Calculate metrics
    results[model_path.replace('.joblib', '')] = {
        'accuracy': accuracy_score(Y_test, Y_pred),
        'precision': precision_score(Y_test, Y_pred, average='weighted'),
        'recall': recall_score(Y_test, Y_pred, average='weighted'),
        'f1_score': f1_score(Y_test, Y_pred, average='weighted')
    }

# Print results
for model_name, metrics in results.items():
    print(f"\n{model_name} Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

# Optional: Save to JSON
import json
with open('evaluation_metrics_svm.json', 'w') as f:
    json.dump(results, f, indent=4)