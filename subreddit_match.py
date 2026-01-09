import os
import pandas as pd
from tqdm import tqdm  

authors_1_path = r""
authors_2_path = r""
brands_path = r""

# subreddit value
brands_df = pd.read_csv(brands_path)
subreddit_list = brands_df['subreddit'].unique()

# dic for subreddit
subreddit_data = {subreddit: [] for subreddit in subreddit_list}

def read_and_process_files(folder_path):
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    
    for i, filename in tqdm(enumerate(all_files), total=len(all_files), desc=f"Processing files in {folder_path}"):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, lineterminator='\n')

        # match
        for subreddit in subreddit_list:
            matching_rows = df[df['subreddit'] == subreddit]
            if not matching_rows.empty:
                subreddit_data[subreddit].append(matching_rows)

# read and match
read_and_process_files(authors_1_path)
read_and_process_files(authors_2_path)


for subreddit, data_list in subreddit_data.items():
    combined_data = pd.concat(data_list, ignore_index=True)
    
    # save subreddit csv
    output_filename = f"{subreddit}_combined.csv"
    output_path = os.path.join(r"", output_filename)
    combined_data.to_csv(output_path, index=False)

print("处理完毕，所有匹配的文件已生成。")
