import argparse
from pathlib import Path
from typing import Optional
import shutil
import pandas as pd
from tqdm import tqdm

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
MODEL_NAME = "bert-base-uncased"


# ---------- Utility ----------
def require_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


# ---------- match ----------
def build_subreddit_index(brands_csv: Path) -> list[str]:
    brands_df = pd.read_csv(brands_csv)
    if "subreddit" not in brands_df.columns:
        raise ValueError(f"{brands_csv} is missing a 'subreddit' column")
    return sorted(brands_df["subreddit"].dropna().unique())


def cmd_match(args: argparse.Namespace) -> None:
    require_exists(args.brands_csv, "Brands CSV")
    input_dirs = [Path(p) for p in args.input_dirs]

    subreddit_list = build_subreddit_index(args.brands_csv)
    subreddit_data: dict[str, list[pd.DataFrame]] = {name: [] for name in subreddit_list}

    for folder in input_dirs:
        require_exists(folder, "Input folder")
        csv_files = sorted(folder.glob(args.pattern))
        if not csv_files:
            print(f"No CSV files found in {folder}, skipping.")
            continue

        for csv_file in tqdm(csv_files, desc=f"Processing {folder.name}", unit="file", ncols=100):
            df = pd.read_csv(csv_file, lineterminator="\n")
            if "subreddit" not in df.columns:
                continue
            filtered = df[df["subreddit"].isin(subreddit_list)]
            for subreddit, group in filtered.groupby("subreddit"):
                subreddit_data[subreddit].append(group)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subreddit, frames in subreddit_data.items():
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        combined.to_csv(args.output_dir / f"{subreddit}_combined.csv", index=False)
    print("处理完毕，所有匹配的文件已生成。")


# ---------- restore ----------
def process_comments(removed_file: Path, combined_file: Path, extracted_file: Path, output_file: Path) -> None:
    all_removed = pd.read_csv(removed_file)
    combined = pd.read_csv(combined_file, lineterminator="\n")
    comments_removed_extracted = pd.read_csv(extracted_file)

    removed_comment_ids = all_removed["comment_id"].tolist()
    extracted_comment_ids = comments_removed_extracted["id"].tolist()

    filtered_combined = combined[
        combined["comment_id"].isin(removed_comment_ids) & ~combined["comment_id"].isin(extracted_comment_ids)
    ]
    filtered_combined_renamed = filtered_combined.rename(
        columns={
            "comment_body": "body",
            "comment_id": "id",
            "post_id": "parent_id",
            "author": "author",
            "subreddit": "subreddit",
            "created_utc": "created_utc",
            "score": "score",
        }
    )

    comments_removed_extracted_after = pd.concat(
        [comments_removed_extracted, filtered_combined_renamed], ignore_index=True
    )
    comments_removed_extracted_after = comments_removed_extracted_after[
        ["body", "id", "parent_id", "subreddit", "author", "score", "created_utc"]
    ]

    output_file.parent.mkdir(parents=True, exist_ok=True)
    comments_removed_extracted_after.to_csv(output_file, index=False, encoding="utf-8")


def cmd_restore(args: argparse.Namespace) -> None:
    require_exists(args.dataset_dir, "Dataset directory")
    subreddit_folders = [folder for folder in args.dataset_dir.iterdir() if folder.is_dir()]

    for subreddit_path in tqdm(subreddit_folders, desc="Processing subreddits"):
        subreddit = subreddit_path.name
        combined_file = subreddit_path / f"{subreddit}_combined.csv"
        rm_file = subreddit_path / f"{subreddit}_all_removed.csv"
        rm_extracted_file = subreddit_path / f"{subreddit}_comments_removed_extracted.csv"
        dlt_file = subreddit_path / f"{subreddit}_all_deleted.csv"
        dlt_extracted_file = subreddit_path / f"{subreddit}_comments_deleted_extracted.csv"

        required_files = [combined_file, rm_file, rm_extracted_file, dlt_file, dlt_extracted_file]
        missing = [p.name for p in required_files if not p.exists()]
        if missing:
            print(f"Skipping {subreddit}: missing files {missing}")
            continue

        subreddit_output_folder = args.output_dir / subreddit
        output_file_1 = subreddit_output_folder / f"{subreddit}_comments_removed_extracted_after.csv"
        output_file_2 = subreddit_output_folder / f"{subreddit}_comments_deleted_extracted_after.csv"

        process_comments(rm_file, combined_file, rm_extracted_file, output_file_1)
        process_comments(dlt_file, combined_file, dlt_extracted_file, output_file_2)

        if not args.skip_copy:
            subreddit_output_folder.mkdir(parents=True, exist_ok=True)
            for file_path in required_files:
                shutil.copy(file_path, subreddit_output_folder / file_path.name)


# ---------- predict ----------
def load_model(model_path: Path, device):
    from transformers import BertTokenizer, BertForSequenceClassification
    import torch

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(LABEL_COLS))
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer


def predict_file(
    csv_path: Path,
    model,
    tokenizer,
    device,
    text_column: str,
    max_len: int,
    batch_size: int,
    output_dir: Path,
) -> Optional[Path]:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import numpy as np

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
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }

    df = pd.read_csv(csv_path, lineterminator="\n")
    if text_column not in df.columns:
        print(f"Skipping {csv_path.name}: missing '{text_column}' column")
        return None

    texts = df[text_column].fillna("").astype(str).values
    dataset = TestCommentDataset(texts, tokenizer, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    predictions = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Predicting {csv_path.name}", unit="batch", ncols=100):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.sigmoid(outputs.logits)
            predictions.append(probs.cpu().numpy())

    if not predictions:
        print(f"No rows found in {csv_path}")
        return None

    import numpy as np

    pred_df = pd.DataFrame(np.concatenate(predictions, axis=0), columns=LABEL_COLS)
    result_df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / csv_path.name.replace(".csv", "_toxic.csv")
    result_df.to_csv(output_path, index=False)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return output_path


def cmd_predict(args: argparse.Namespace) -> None:
    import torch

    require_exists(args.model_path, "Model path")
    require_exists(args.input_dir, "Input directory")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.model_path, device)

    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

    for csv_file in csv_files:
        output_path = predict_file(
            csv_file,
            model,
            tokenizer,
            device,
            text_column=args.text_column,
            max_len=args.max_length,
            batch_size=args.batch_size,
            output_dir=args.output_dir,
        )
        if output_path:
            print(f"Processed: {output_path}")
    print("Processing complete.")


# ---------- rename ----------
def cmd_rename(args: argparse.Namespace) -> None:
    require_exists(args.input_dir, "Input directory")
    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        for i, new_col in enumerate(LABEL_COLS):
            old_col = f"label_{i}"
            if old_col in df.columns:
                df.rename(columns={old_col: new_col}, inplace=True)
        output_path = args.output_dir / csv_file.name
        df.to_csv(output_path, index=False)
        print(f"Updated file saved: {output_path}")
    print("Label renaming complete.")


# ---------- sentiment ----------
def cmd_sentiment(args: argparse.Namespace) -> None:
    import torch
    from transformers import pipeline as hf_pipeline, AutoTokenizer

    require_exists(args.input_dir, "Input directory")
    csv_files = sorted(args.input_dir.glob(args.pattern))
    if not csv_files:
        raise FileNotFoundError(f"No files found in {args.input_dir} matching {args.pattern}")

    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    classifier = hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        tokenizer=tokenizer,
        device=device,
    )

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, lineterminator="\n")
        if args.text_column not in df.columns:
            print(f"Skipping {csv_file.name}: missing '{args.text_column}' column")
            continue

        for i in tqdm(range(len(df)), desc=f"Processing {csv_file.name}", ncols=100):
            text = str(df.loc[i, args.text_column])
            if text.strip() == "":
                continue
            encoding = tokenizer(text, truncation=True, padding=False, max_length=args.max_seq_len)
            truncated_text = tokenizer.decode(encoding["input_ids"], skip_special_tokens=True)

            sentiment = classifier(truncated_text)[0]
            df.loc[i, "Sentiment_Label"] = 1 if sentiment["label"] == "POSITIVE" else -1
            df.loc[i, "Sentiment_Score"] = round(sentiment["score"], 4)
            df.loc[i, "Sentiment_Score_Final"] = df.loc[i, "Sentiment_Label"] * df.loc[i, "Sentiment_Score"]

        base_name = csv_file.name.replace(args.strip_suffix, "") if args.strip_suffix else csv_file.name
        output_name = base_name.replace(".csv", f"{args.output_suffix}.csv")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / output_name
        df.to_csv(output_path, index=False)
        print(f"Processed file saved to {output_path}")


# ---------- parser ----------
def add_match_parser(subparsers):
    parser = subparsers.add_parser("match", help="Aggregate rows per subreddit across author CSV folders.")
    parser.add_argument("--input-dirs", nargs="+", required=True, type=Path, help="Folders containing author CSVs.")
    parser.add_argument("--brands-csv", required=True, type=Path, help="CSV with 'subreddit' column.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write <subreddit>_combined.csv.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input CSVs. Default: *.csv")
    parser.set_defaults(func=cmd_match)


def add_restore_parser(subparsers):
    parser = subparsers.add_parser("restore", help="Add removed/deleted comments back into extracted datasets.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("Dataset_Add"), help="Root folder with subreddit CSVs.")
    parser.add_argument("--output-dir", type=Path, default=Path("Dataset_Add"), help="Folder to save *_after.csv files.")
    parser.add_argument("--skip-copy", action="store_true", help="Do not copy original CSVs into output directory.")
    parser.set_defaults(func=cmd_restore)


def add_predict_parser(subparsers):
    parser = subparsers.add_parser("predict", help="Run multi-label toxicity prediction.")
    parser.add_argument("--model-path", required=True, type=Path, help="Path to fine-tuned BERT state dict (.bin).")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing CSVs to score.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Where to write *_toxic.csv files.")
    parser.add_argument("--text-column", default="body", help="Column containing text to classify. Default: body")
    parser.add_argument("--max-length", type=int, default=128, help="Maximum sequence length for tokenizer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for inference.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for input CSVs. Default: *.csv")
    parser.set_defaults(func=cmd_predict)


def add_rename_parser(subparsers):
    parser = subparsers.add_parser("rename-labels", help="Rename label_0..5 columns to descriptive toxicity labels.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Folder containing prediction CSVs.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Folder to write renamed CSVs.")
    parser.add_argument("--pattern", default="*_toxic.csv", help="Glob pattern for files. Default: *_toxic.csv")
    parser.set_defaults(func=cmd_rename)


def add_sentiment_parser(subparsers):
    parser = subparsers.add_parser("sentiment", help="Run sentiment analysis on Reddit comment CSVs.")
    parser.add_argument("--input-dir", required=True, type=Path, help="Directory containing input CSV files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to store sentiment outputs.")
    parser.add_argument("--text-column", default="body", help="Column with text to analyze. Default: body")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Maximum token length for truncation.")
    parser.add_argument("--strip-suffix", default="_after_toxic", help="Suffix to remove before saving.")
    parser.add_argument("--output-suffix", default="_done", help="Suffix to append before .csv when saving.")
    parser.add_argument("--pattern", default="*.csv", help="Glob pattern for files to process.")
    parser.set_defaults(func=cmd_sentiment)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reddit Toxic pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_match_parser(subparsers)
    add_restore_parser(subparsers)
    add_predict_parser(subparsers)
    add_rename_parser(subparsers)
    add_sentiment_parser(subparsers)
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
