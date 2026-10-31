"""Крок 8.3: готує вибірки Rotten Tomatoes і завантажує їх у S3.

Ті самі 2000 / 500 прикладів, що у фазі 5 (train/finetune.py, seed=42).
У S3 з'являються дві папки — канали train і validation для training job.
Запуск з кореня проєкту: python aws/upload_data.py
"""
from pathlib import Path

import boto3
from datasets import load_dataset

from settings import BUCKET, DATA_S3_PREFIX, REGION

OUT_DIR = Path("build/data")  # build/ уже в .gitignore

# 1. Вибірки — рівно як у finetune.py.
ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
splits = {
    "train": ds["train"].shuffle(seed=42).select(range(2000)),
    "validation": ds["validation"].shuffle(seed=42).select(range(500)),
}

s3 = boto3.Session(region_name=REGION).client("s3")

for channel, data in splits.items():
    # 2. Локальна копія у форматі JSONL: один рядок = {"text": "...", "label": 0 або 1}.
    local_path = OUT_DIR / channel / f"{channel}.jsonl"
    local_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_json(str(local_path))

    # 3. Завантаження в S3: кожен канал — своя папка.
    key = f"{DATA_S3_PREFIX}/{channel}/{channel}.jsonl"
    s3.upload_file(str(local_path), BUCKET, key)

    positive = sum(data["label"])
    print(f"{channel}: {len(data)} прикладів, позитивних {positive} "
          f"-> s3://{BUCKET}/{key}")

print("Готово. Перевірка: aws s3 ls s3://" + BUCKET + "/" + DATA_S3_PREFIX + "/ --recursive")