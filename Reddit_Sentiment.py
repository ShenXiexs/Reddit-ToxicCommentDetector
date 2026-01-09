from transformers import pipeline, AutoTokenizer
classifier = pipeline("sentiment-analysis")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
import pandas as pd
import os
from tqdm import tqdm

input_folder = ''
output_folder = ''


MAX_SEQ_LEN = 512


for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):
        
        file_path = os.path.join(input_folder, file_name)
        df = pd.read_csv(file_path, lineterminator='\n')
        
        for i in tqdm(range(len(df)), desc=f"Processing {file_name}", ncols=100):
            text = str(df.loc[i, 'body'])  
            
            if text.strip() == "":
                continue
            
            encoding = tokenizer(text, truncation=True, padding=False, max_length=MAX_SEQ_LEN)
            truncated_text = tokenizer.decode(encoding['input_ids'], skip_special_tokens=True)
            
            sentiment = classifier(truncated_text)[0]
            df.loc[i, 'Sentiment_Label'] = 1 if sentiment['label'] == 'POSITIVE' else -1
            df.loc[i, 'Sentiment_Score'] = round(sentiment['score'], 4)
            df.loc[i, 'Sentiment_Score_Final'] = df.loc[i, 'Sentiment_Label'] * df.loc[i, 'Sentiment_Score']
        
        new_file_name = file_name.replace('_after_toxic', '')
        new_file_name = new_file_name.replace('.csv', '_done.csv')
        new_file_path = os.path.join(output_folder, new_file_name)
        
        df.to_csv(new_file_path, index=False)
        
        print(f"Processed file saved to {new_file_path}")