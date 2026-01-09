# Reddit Toxic Pipeline

Single-entry CLI to gather subreddit comments, restore removed/deleted items, run toxicity scoring, rename labels, and perform sentiment analysis.

## Quick Start
```bash
pip install -r requirements.txt
# First model download requires internet access (Hugging Face).
```

## Commands
- `match`: merge rows per subreddit from author CSV folders.
  ```bash
  python pipeline.py match \
    --input-dirs data/authors_part1 data/authors_part2 \
    --brands-csv data/brands.csv \
    --output-dir Dataset_Add/raw_combined
  ```
- `restore`: add removed/deleted comments back and export `*_after.csv`.
  ```bash
  python pipeline.py restore \
    --dataset-dir Dataset_Add/raw_combined \
    --output-dir Dataset_Add \
    --skip-copy   # optional: skip copying originals into output
  ```
- `predict`: run multi-label toxicity inference (requires fine-tuned BERT state dict).
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
- `sentiment`: sentiment analysis with DistilBERT SST-2.
  ```bash
  python pipeline.py sentiment \
    --input-dir predictions \
    --output-dir sentiment \
    --text-column body \
    --strip-suffix _after_toxic \
    --output-suffix _done
  ```

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
- GPU speeds up prediction and sentiment; CUDA is auto-detected.
- Ensure the text column name matches your data (default `body`); override with CLI flags.
- The original notebooks (`*.ipynb`) remain for reference/training; pipeline covers the production flow. If you want a slimmer repo, you can remove unused notebooks after archiving them elsewhere.
