# Reddit Toxic Pipeline

Toolkit for collecting Reddit brand data, restoring removed/deleted comments, training a toxicity classifier, and running inference/sentiment analysis through a single CLI.

## Setup
- Python 3.9+ recommended.
- Install dependencies: `pip install -r requirements.txt`.
- First run will download Hugging Face models (requires internet once).
- For data collection with the Reddit API, add your own `client_id`, `client_secret`, and `user_agent` in `Reddit Pushshift.py`.

## Workflow Overview
### 1) Collect submissions/comments
- Export submission IDs from Pushshift into `<brand>_submissions_pushshift.csv` files under `input_dir`.
- Fill `client_id/client_secret/user_agent` plus `input_dir` in `Reddit Pushshift.py`, then run it to fetch metadata via PRAW and save `<brand>_submissions_redditapi1.csv`.

### 2) Tag and clean datasets
- `RedditProcess0612.py` step 1 (set `root_path`): for each brand folder, reads `<brand>_all_deleted.csv`, `<brand>_all_removed.csv`, and `<brand>_submissions_redditapi.csv`, then writes `_parent.csv` files with `parent_deleted/parent_removed` flags.
- `RedditProcess0612.py` step 2 (set `base_dir`): adds `brand_relevant` flag to `<brand>_comments_pushshift_parent.csv` using brand-specific regex rules.

### 3) Train toxicity classifier (optional if you have a model already)
- `Reddit_Toxic.py` fine-tunes `bert-base-uncased` on the Kaggle toxic comments dataset; adjust dataset paths near the top.
- The best checkpoint is saved to `best_model_state.bin` and can be used by the CLI `predict` command.

### 4) Production CLI (pipeline.py)
- `match`: merge rows per subreddit across author CSV folders.
  ```bash
  python pipeline.py match \
    --input-dirs data/authors_part1 data/authors_part2 \
    --brands-csv data/brands.csv \
    --output-dir Dataset_Add/raw_combined
  ```
- `restore`: reattach removed/deleted comments and export `*_after.csv`.
  ```bash
  python pipeline.py restore \
    --dataset-dir Dataset_Add/raw_combined \
    --output-dir Dataset_Add \
    --skip-copy   # optional: avoid copying originals into output
  ```
- `predict`: run multi-label toxicity inference (needs fine-tuned BERT state dict).
  ```bash
  python pipeline.py predict \
    --model-path models/toxic_model.bin \
    --input-dir Dataset_Add/<subreddit_folder> \
    --output-dir predictions \
    --text-column body \
    --max-length 128 \
    --batch-size 8
  ```
- `rename-labels`: rename `label_0..5` to readable toxicity labels if needed.
  ```bash
  python pipeline.py rename-labels \
    --input-dir predictions \
    --output-dir predictions
  ```
- `sentiment`: run SST-2 sentiment scoring.
  ```bash
  python pipeline.py sentiment \
    --input-dir predictions \
    --output-dir sentiment \
    --text-column body \
    --strip-suffix _after_toxic \
    --output-suffix _done
  ```

### 5) Legacy notebook scripts (converted to .py)
- `Reddit_Sentiment.py`: standalone sentiment pipeline similar to the CLI command.
- `Reddit Pushshift.py`: PRAW-based submission metadata fetcher for Pushshift exports.
- `RedditProcess0612.py`: deletion/brand relevance tagging utilities.
- `Reddit_Toxic.py`: full training loop and inference export for the toxicity model.

## Expected Layout (example)
```
Reddit-Toxic-Code/
├─ data/
│  ├─ brands.csv                # column: subreddit
│  └─ authors_part*/            # author CSV exports
├─ Dataset_Add/
│  └─ <subreddit>/
│     ├─ <subreddit>_combined.csv
│     ├─ <subreddit>_all_removed.csv
│     ├─ <subreddit>_comments_removed_extracted.csv
│     ├─ <subreddit>_all_deleted.csv
│     └─ <subreddit>_comments_deleted_extracted.csv
└─ models/
   └─ toxic_model.bin           # fine-tuned BERT state dict
```

## Notes
- GPU will speed up `predict` and `sentiment`; CUDA is auto-detected.
- Ensure the text column name matches your data (default `body`); override with CLI flags.
- Converted `.py` files mirror the original notebooks for transparency and offline editing.
