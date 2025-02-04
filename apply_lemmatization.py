import pandas as pd
import stanza
from tqdm import tqdm

# Step 1: Download and initialize the Stanza Hindi pipeline with GPU support
stanza.download('hi')  # Download Hindi model 
nlp = stanza.Pipeline('hi', processors='tokenize,pos,lemma', use_gpu=True, verbose=False)

# Step 2: Load the dataset
file_path = '/Users/ayushidubey/Desktop/Research work/combined_and_cleaned_data/final_dataset.csv'  # Replace with your actual path
df = pd.read_csv(file_path)

# Ensure the dataset has the required 'content' column
if 'content' not in df.columns:
    raise ValueError("The dataset must have a 'content' column containing Hindi text.")

# Step 3: Lemmatization function using Stanza
def lemmatize_text(text):
    try:
        # Process the text using Stanza
        doc = nlp(text)
        # Extract lemmas from the processed tokens
        lemmas = " ".join([word.lemma for sentence in doc.sentences for word in sentence.words])
        return lemmas
    except Exception as e:
        print(f"Error processing text: {text}. Error: {e}")
        return text  # Return original text if processing fails

# Step 4: Apply lemmatization to the dataset with tqdm
print("Applying lemmatization to the dataset. This may take some time...")
tqdm.pandas()  # Enable tqdm progress bar for pandas
df['lemmatized_content'] = df['content'].progress_apply(lemmatize_text)

# Step 5: Save the updated dataset
output_file = '/Users/ayushidubey/Desktop/Research work/combined_and_cleaned_data/final_lemmatized_dataset.csv'
df.to_csv(output_file, index=False)

print(f"Lemmatization complete. Processed data saved to {output_file}")