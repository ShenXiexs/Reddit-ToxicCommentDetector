# Reddit Toxic Code

工具链用于收集特定子版块的 Reddit 评论、补齐被删内容，并对文本做多标签“有害”识别与情感分析。核心入口整合为单一 CLI：`python pipeline.py <command> ...`。

## 概览 / Overview
- 入口 CLI：`pipeline.py` 提供子命令 `match`、`restore`、`predict`、`rename-labels`、`sentiment`，覆盖原分散脚本功能（同名旧脚本仅保留轻量包装）。
- 爬取：`Reddit Pushshift.ipynb`、`RedditAPI.Rmd`/`Reddit_Scraping.Rmd` 处理数据抓取。
- 模型：`Reddit_Toxic.ipynb` 训练多标签 BERT；`Reddit_Merge.Rmd`、`RedditProcess0612.ipynb` 辅助处理与检查。
- 其它数据整合：`Reddit_Merge.Rmd`、`RedditProcess0612.ipynb` 辅助处理与检查。

## 环境 / Setup
- Python 3.9+ 建议使用虚拟环境。
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  # 如需跑 Notebook，额外安装：pip install jupyter
  ```
- 首次运行 Hugging Face 模型需联网下载；已下载可离线使用。

## 数据布局示例 / Expected Layout
```
Reddit-Toxic-Code/
├─ data/
│  ├─ brands.csv                # 包含目标子版块列表的表（字段：subreddit）
│  └─ authors_part*/            # 从 Pushshift/PRAW 导出的作者 CSV
├─ Dataset_Add/
│  └─ <subreddit>/
│     ├─ <subreddit>_combined.csv
│     ├─ <subreddit>_all_removed.csv
│     ├─ <subreddit>_comments_removed_extracted.csv
│     ├─ <subreddit>_all_deleted.csv
│     └─ <subreddit>_comments_deleted_extracted.csv
└─ models/
   └─ toxic_model.bin           # 训练好的 BERT state dict
```

 ## 处理流程 / Workflow
1) **按子版块聚合**  
   ```bash
   python pipeline.py match \
     --input-dirs data/authors_part1 data/authors_part2 \
     --brands-csv data/brands.csv \
     --output-dir Dataset_Add/raw_combined
   ```

2) **补回 removed/deleted 评论并导出 *_after.csv**  
   ```bash
   python pipeline.py restore \
     --dataset-dir Dataset_Add/raw_combined \
     --output-dir Dataset_Add \
     # 如不想拷贝原始文件到输出目录，追加 --skip-copy
   ```

3) **多标签有害性预测**（需训练好的模型）  
   ```bash
   python pipeline.py predict \
     --model-path models/toxic_model.bin \
     --input-dir Dataset_Add/<subreddit_folder> \
     --output-dir predictions \
     --text-column body \
     --max-length 128 \
     --batch-size 8
   ```
   输出会生成 `<name>_toxic.csv`，包含六个概率列：toxic、severe_toxic、obscene、threat、insult、identity_hate。

4) **统一列名（如模型输出仍是 label_0…5）**  
   ```bash
   python pipeline.py rename-labels --input-dir predictions --output-dir predictions
   ```

5) **情感分析**  
   ```bash
   python pipeline.py sentiment \
     --input-dir predictions \
     --output-dir sentiment \
     --text-column body \
     --strip-suffix _after_toxic \
     --output-suffix _done
   ```
   结果增加 `Sentiment_Label`（1/-1）、`Sentiment_Score`、`Sentiment_Score_Final`。

## 模型训练 / Model Training
- `Reddit_Toxic.ipynb` 使用 Kaggle Wiki Toxic 评论数据微调 `bert-base-uncased`（6 标签）。训练完导出 `state_dict` 为 `toxic_model.bin` 供 `Reddit_Predict.py` 调用。
- 可根据需要调整标签映射或超参数后重新导出。

## 提示 / Notes
- GPU 可显著加速推理与情感分析；脚本会自动检测可用的 CUDA。
- 运行前确认路径存在且列名一致（文本列默认 `body`）；可用脚本参数覆盖。
- 处理大批量 CSV 时建议分批/分目录执行，避免一次性占用过多内存。
