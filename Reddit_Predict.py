import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
import numpy as np

# Paths
model_path = '/Users/samxie/Research/HEC/Reddit Toxic 0412/best_model_state.bin'
input_dir = '/Users/samxie/Research/HEC/Reddit Toxic 0412/Data_for_Predict2'
output_dir = '/Users/samxie/Research/HEC/Reddit Toxic 0412/Data_After_Predict'

# Load the trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Dataset class for test data
class TestCommentDataset(Dataset):
    def __init__(self, comments, tokenizer, max_len):
        self.comments = comments
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, item):
        comment = str(self.comments[item])

        encoding = self.tokenizer.encode_plus(
            comment,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }

# Function to predict labels for the 'body' column
def predict_labels(csv_file, pbar):
    # Set the dtype for specific columns
    dtype_dict = {2: str, 3: str, 4: str}
    
    df = pd.read_csv(csv_file, dtype=dtype_dict, lineterminator='\n')
    
    if 'body' not in df.columns:
        return None  # Skip files that don't have 'body' column

    # Tokenizing the 'body' column
    test_texts = df['body'].values
    test_dataset = TestCommentDataset(test_texts, tokenizer, max_len=128)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # Generate predictions
    predictions = []
    model.eval()
    print(f"\nGenerating predictions for {os.path.basename(csv_file)}...")

    # Track progress within each CSV file
    total_rows = len(df)
    processed_rows = 0

    # Start processing and updating file-level progress bar
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.sigmoid(logits)  # Convert logits to probabilities (0-1 range)
            predictions.append(probs.cpu().numpy())

            processed_rows += len(batch['input_ids'])

            # Calculate the percentage of rows processed and update progress bar
            pbar.set_postfix(processed=processed_rows, total=total_rows)
            pbar.update(len(batch['input_ids']))

    # Concatenate the predictions
    predictions = np.concatenate(predictions, axis=0)
    
    # Labels corresponding to each prediction
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    pred_df = pd.DataFrame(predictions, columns=label_cols)

    # Keep original columns and add predictions
    result_df = pd.concat([df, pred_df], axis=1)

    # Save the updated dataframe
    updated_filename = os.path.join(output_dir, os.path.basename(csv_file).replace('.csv', '_toxic.csv'))
    result_df.to_csv(updated_filename, index=False)

    # Clear cache after processing each file
    del result_df, predictions  # Delete dataframes to free memory
    torch.cuda.empty_cache()  # Release GPU memory

    return updated_filename

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List all CSV files in the input directory
csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
total_files = len(csv_files)

# Process each file sequentially and track progress
with tqdm(total=total_files, desc="Processing files", unit="file", ncols=100) as pbar:
    for csv_file in csv_files:
        csv_path = os.path.join(input_dir, csv_file)
        updated_file = predict_labels(csv_path, pbar)
        if updated_file:
            tqdm.write(f"Processed: {updated_file}")

print("Processing complete.")                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
