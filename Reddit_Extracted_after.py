import os
import pandas as pd
from tqdm import tqdm
import shutil

def process_comments(removed_file, combined_file, extracted_file, output_file):
    all_removed = pd.read_csv(removed_file)
    combined = pd.read_csv(combined_file, lineterminator='\n')
    comments_removed_extracted = pd.read_csv(extracted_file)
    removed_comment_ids = all_removed['comment_id'].tolist()
    extracted_comment_ids = comments_removed_extracted['id'].tolist()
    filtered_combined = combined[combined['comment_id'].isin(removed_comment_ids) & ~combined['comment_id'].isin(extracted_comment_ids)]
    filtered_combined_renamed = filtered_combined.rename(columns={
        'comment_body': 'body',
        'comment_id': 'id',
        'post_id': 'parent_id',
        'author': 'author',
        'subreddit': 'subreddit',
        'created_utc': 'created_utc',
        'score': 'score'
    })

    comments_removed_extracted_after = pd.concat([comments_removed_extracted, filtered_combined_renamed], ignore_index=True)
    comments_removed_extracted_after = comments_removed_extracted_after[['body', 'id', 'parent_id', 'subreddit', 'author', 'score', 'created_utc']]
    comments_removed_extracted_after.to_csv(output_file, index=False, encoding='utf-8')


def process_all_subreddits(dataset_dir, output_dir):
    subreddit_folders = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]

    for subreddit in tqdm(subreddit_folders, desc="Processing subreddits"):
        subreddit_path = os.path.join(dataset_dir, subreddit)
        combined_file = os.path.join(subreddit_path, f'{subreddit}_combined.csv')
        rm_file = os.path.join(subreddit_path, f'{subreddit}_all_removed.csv')
        rm_extracted_file = os.path.join(subreddit_path, f'{subreddit}_comments_removed_extracted.csv')
        dlt_file = os.path.join(subreddit_path, f'{subreddit}_all_deleted.csv')
        dlt_extracted_file = os.path.join(subreddit_path, f'{subreddit}_comments_deleted_extracted.csv')
        
        # Create a new folder for the subreddit inside Dataset_Add
        subreddit_output_folder = os.path.join(output_dir, subreddit)
        os.makedirs(subreddit_output_folder, exist_ok=True)

        # Process and save after CSV files in the new folder
        output_file_1 = os.path.join(subreddit_output_folder, f'{subreddit}_comments_removed_extracted_after.csv')
        output_file_2 = os.path.join(subreddit_output_folder, f'{subreddit}_comments_deleted_extracted_after.csv')
        
        process_comments(rm_file, combined_file, rm_extracted_file, output_file_1)
        process_comments(dlt_file, combined_file, dlt_extracted_file, output_file_2)
        
        # Move all the original CSV files to the new folder
        for file_name in [f'{subreddit}_combined.csv', f'{subreddit}_all_removed.csv', f'{subreddit}_comments_removed_extracted.csv', 
                          f'{subreddit}_all_deleted.csv', f'{subreddit}_comments_deleted_extracted.csv']:
            original_file = os.path.join(subreddit_path, file_name)
            if os.path.exists(original_file):
                shutil.copy(original_file, os.path.join(subreddit_output_folder, file_name))


# Call the function to process all subreddits and save output in 'Dataset_Add'
process_all_subreddits('Dataset_Add', 'Dataset_Add')
