"""Крок 8.3: скрипт тренування, який виконується ВСЕРЕДИНІ training job SageMaker.

Основа — train/finetune.py з фази 5. Відмінності:
  - дані беруться з каналів SageMaker (SM_CHANNEL_TRAIN, SM_CHANNEL_VALIDATION),
    а не з Hugging Face Hub;
  - гіперпараметри приходять аргументами командного рядка (--epochs 1 ...);
  - модель і токенізатор зберігаються в SM_MODEL_DIR, SageMaker сам пакує їх у model.tar.gz;
  - замість MLflow метрики друкуються рядками name=value (їх виловить CloudWatch).

Локальна перевірка з кореня проєкту (кілька десятків прикладів, ~1 хвилина):
    python train/sm_job/sm_train.py --limit 64
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments, pipeline, set_seed)


# 1. Аргументи: гіперпараметри і шляхи.
def parse_args():
    p = argparse.ArgumentParser()
    # Гіперпараметри. У job їх передасть train_job.py; тут — значення як у фазі 5.
    p.add_argument("--model-name", default="distilbert/distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0,
                   help="0 = усі приклади; >0 = лише для швидкої локальної перевірки")
    # Шляхи. У контейнері їх дають змінні середовища SageMaker;
    # локально таких змінних немає, і береться друге значення.
    p.add_argument("--train-dir",
                   default=os.environ.get("SM_CHANNEL_TRAIN", "build/data/train"))
    p.add_argument("--validation-dir",
                   default=os.environ.get("SM_CHANNEL_VALIDATION", "build/data/validation"))
    p.add_argument("--model-dir",
                   default=os.environ.get("SM_MODEL_DIR", "build/sm_model"))
    p.add_argument("--output-dir",
                   default=os.environ.get("SM_OUTPUT_DATA_DIR", "build/sm_output"))
    return p.parse_args()


# 2. Дані: усі *.jsonl з папки каналу -> список словників {"text": ..., "label": ...}.
def read_jsonl(folder, limit):
    rows = []
    for path in sorted(Path(folder).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    if not rows:
        raise FileNotFoundError(f"У папці {folder} немає файлів *.jsonl")
    return rows[:limit] if limit > 0 else rows


# 3. Dataset для Trainer: токенізуємо один раз, віддаємо приклади за номером.
class ReviewsDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_length):
        enc = tok([r["text"] for r in rows], truncation=True,
                  padding="max_length", max_length=max_length)
        # Лише ті входи, які приймає DistilBERT (без token_type_ids).
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor(self.input_ids[i]),
            "attention_mask": torch.tensor(self.attention_mask[i]),
            "labels": torch.tensor(self.labels[i]),
        }


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)}


def main():
    args = parse_args()
    print("transformers", transformers.__version__, "| torch", torch.__version__)
    print("Аргументи:", vars(args))
    set_seed(args.seed)

    # 4. Читаємо дані з каналів.
    train_rows = read_jsonl(args.train_dir, args.limit)
    val_rows = read_jsonl(args.validation_dir, args.limit)
    print(f"Дані: train={len(train_rows)}, validation={len(val_rows)}")

    # 5. Токенізатор і модель — як у finetune.py.
    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = ReviewsDataset(train_rows, tok, args.max_length)
    val_ds = ReviewsDataset(val_rows, tok, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    # 6. Тренування.
    training_args = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "trainer"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
        seed=args.seed,
    )
    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    train_result = trainer.train()
    metrics = trainer.evaluate()
    metrics["train_loss"] = train_result.training_loss

    # 7. Метрики: рядки name=value для CloudWatch і файл metrics.json.
    for name in ["eval_accuracy", "eval_f1", "eval_loss", "train_loss"]:
        print(f"{name}={metrics[name]:.4f}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 8. Зберігаємо модель і токенізатор туди, звідки SageMaker забере їх у S3.
    model_dir = Path(args.model_dir)
    model.save_pretrained(model_dir)
    tok.save_pretrained(model_dir)

    # 9. Сумісність з контейнером інференсу (transformers 4.51.3) — як блок 2а
    #    у package_model.py. У контейнері нічого не змінить, а локально на
    #    transformers 5 виправить клас токенізатора.
    if model.config.model_type == "distilbert":
        cfg_path = model_dir / "tokenizer_config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["tokenizer_class"] = "DistilBertTokenizer"
        cfg["model_input_names"] = ["input_ids", "attention_mask"]
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    # 10. Перевірка: модель завантажується з папки і відповідає.
    clf = pipeline("text-classification", model=str(model_dir))
    assert "token_type_ids" not in clf.tokenizer("test"), "токенізатор видає token_type_ids"
    for text in ["This is the best movie I have ever seen",
                 "This was a terrible, boring waste of time"]:
        print(f"{text!r} -> {clf(text)[0]}")
    print("Файли моделі:", sorted(p.name for p in model_dir.iterdir()))


if __name__ == "__main__":
    main()