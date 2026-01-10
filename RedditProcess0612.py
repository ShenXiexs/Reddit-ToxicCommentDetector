# %%
import pandas as pd
import os
from tqdm import tqdm

# Root directory containing all brand folders
root_path = ""

# Function to process a single brand folder
def process_brand_folder(brand_folder):
    brand_name = os.path.basename(brand_folder)

    # Construct reference file paths
    deleted_path = os.path.join(brand_folder, f"{brand_name}_all_deleted.csv")
    removed_path = os.path.join(brand_folder, f"{brand_name}_all_removed.csv")
    submissions_path = os.path.join(brand_folder, f"{brand_name}_submissions_redditapi.csv")

    # Skip if any of the required reference files are missing
    if not (os.path.exists(deleted_path) and os.path.exists(removed_path) and os.path.exists(submissions_path)):
        print(f"Skipping {brand_name}: missing reference files")
        return

    try:
        # Load deleted and removed comment IDs
        deleted_ids = set(pd.read_csv(deleted_path, low_memory=False)['comment_id'].astype(str))
        removed_ids = set(pd.read_csv(removed_path, low_memory=False)['comment_id'].astype(str))

        # Load submissions and extract IDs based on title content
        submissions_df = pd.read_csv(submissions_path, low_memory=False)
        submissions_df['id'] = submissions_df['id'].astype(str)
        submissions_df['title'] = submissions_df['title'].astype(str)

        deleted_title_ids = set(submissions_df[submissions_df['title'].str.contains(r"\[deleted by user\]", case=False, na=False)]['id'])
        removed_title_ids = set(submissions_df[submissions_df['title'].str.contains("Removed by Reddit", case=False, na=False)]['id'])

        # Target comment files to process
        target_files = [
            f"{brand_name}_comments_deleted_extracted.csv",
            f"{brand_name}_comments_removed_extracted.csv",
            f"{brand_name}_comments_pushshift.csv"
        ]

        for filename in target_files:
            file_path = os.path.join(brand_folder, filename)
            if not os.path.exists(file_path):
                print(f"File not found: {filename}, skipping")
                continue

            # Load comment file with type warning suppression
            df = pd.read_csv(file_path, engine='python', encoding='utf-8')

            # Extract the actual parent ID (strip prefix like "t1_", "t3_")
            df['clean_parent_id'] = df['parent_id'].astype(str).apply(lambda x: x.split('_')[-1])

            # Initialize progress bar for row-wise application
            tqdm.pandas(desc=f"Processing {filename}")
            df['parent_deleted'] = df['clean_parent_id'].progress_apply(
                lambda x: 1 if x in deleted_ids or x in deleted_title_ids else 0
            )
            df['parent_removed'] = df['clean_parent_id'].progress_apply(
                lambda x: 1 if x in removed_ids or x in removed_title_ids else 0
            )

            # Drop temporary column
            df.drop(columns=['clean_parent_id'], inplace=True)

            # Save the processed file with _parent suffix
            output_file = filename.replace(".csv", "_parent.csv")
            output_path = os.path.join(brand_folder, output_file)
            df.to_csv(output_path, index=False)
            print(f"Processed and saved: {output_file}")

    except Exception as e:
        print(f"Error processing {brand_name}: {e}")

# Iterate through all subfolders in the root directory
for folder in os.listdir(root_path):
    full_path = os.path.join(root_path, folder)
    if os.path.isdir(full_path):
        process_brand_folder(full_path)

# %%
import pandas as pd
import os
from tqdm import tqdm

# Path
base_dir = "/Users/samxie/Research/HEC/Reddit Process 0610/Focal Data Done"

# Rules
brand_regex_dict = {
    "AirBnB": "AirBnB",
    "amazon": "amazon",
    "Apple": "Apple",
    "Blizzard": "Blizzard",
    "BMW": "BMW",
    "Dominos": "Dominos",
    "EASportsUFC": "EASportsUFC|EASports|UFC|EA",
    "Ebay": "Ebay",
    "glossier": "glossier",
    "intel": "intel",
    "McDonalds": "McDonalds",
    "microsoft": "microsoft|ms",
    "mintmobile": "mintmobile",
    "netflix": "netflix",
    "NintendoSwitch": "NintendoSwitch|Nintendo|Switch",
    "nvidia": "nvidia",
    "paypal": "paypal",
    "peloton": "peloton",
    "playstation": "playstation|ps4|ps5",
    "QuickBooks": "QuickBooks",
    "razer": "razer",
    "RobinHood": "RobinHood",
    "spotify": "spotify",
    "starbucks": "starbucks|starbuck",
    "Steam": "Steam",
    "supremeclothing": "supremeclothing|supreme",
    "tacobell": "tacobell",
    "teslamotors": "teslamotors|tesla",
    "Toyota": "Toyota",
    "xbox": "xbox"
}

def extract_id(pid):
    return pid.split("_")[1] if isinstance(pid, str) and "_" in pid else None

def check_brand_relevance(row, brand_regex, id_to_body, id_to_parent):
    import re
    pattern = re.compile(brand_regex, re.IGNORECASE)
    if pattern.search(row['body']):
        return 1
    visited = set()
    parent_id = extract_id(row['parent_id'])
    while parent_id and parent_id not in visited:
        visited.add(parent_id)
        parent_body = id_to_body.get(parent_id, "")
        if pattern.search(parent_body):
            return 1
        parent_id = extract_id(id_to_parent.get(parent_id))
    return 0

for brand, regex in brand_regex_dict.items():
    folder = os.path.join(base_dir, brand)
    filename = f"{brand}_comments_pushshift_parent.csv"
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print(f"File not exist: {filepath}")
        continue

    print(f"Processing brand: {brand}")
    df = pd.read_csv(filepath, dtype=str)
    df['body'] = df['body'].fillna("").str.lower()
    df['id'] = df['id'].astype(str)
    df['parent_id'] = df['parent_id'].astype(str)

    id_to_body = df.set_index('id')['body'].to_dict()
    id_to_parent = df.set_index('id')['parent_id'].to_dict()

    tqdm.pandas(desc=f"Brand: {brand}")
    df['brand_relevant'] = df.progress_apply(
        lambda row: check_brand_relevance(row, regex, id_to_body, id_to_parent), axis=1
    )

    output_path = os.path.join(folder, f"{brand}_comments_pushshift_parent_with_brand_relevant.csv")
    df.to_csv(output_path, index=False)
    print(f"Saved to：{output_path}")
