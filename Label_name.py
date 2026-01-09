import os
import pandas as pd

# Paths
output_dir = ''

# New label columns
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# List all CSV files in the output directory
csv_files = [f for f in os.listdir(output_dir) if f.endswith('_toxic.csv')]

# Function to rename label columns
def rename_label_columns(csv_file):
    df = pd.read_csv(csv_file)

    # Ensure we are renaming the correct columns
    for i, new_col in enumerate(label_cols):
        old_col = f'label_{i}'
        if old_col in df.columns:
            df.rename(columns={old_col: new_col}, inplace=True)

    # Save the updated dataframe with new column names
    updated_filename = os.path.join(output_dir, os.path.basename(csv_file))
    df.to_csv(updated_filename, index=False)

    print(f"Updated file saved: {updated_filename}")

# Process all CSV files in the output directory
for csv_file in csv_files:
    csv_file_path = os.path.join(output_dir, csv_file)
    rename_label_columns(csv_file_path)

print("Label renaming complete.")
