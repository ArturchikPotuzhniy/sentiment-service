import accelerate
import torch
import transformers
import mlflow
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)

BASE = "distilbert/distilbert-base-uncased"

ds = load_dataset("cornell-movie-review-data/rotten_tomatoes")
train = ds["train"].shuffle(seed=42).select(range(2000))
val = ds["validation"].shuffle(seed=42).select(range(500))

tok = AutoTokenizer.from_pretrained(BASE)

def tokenize(b):
    return tok(b["text"], truncation=True, padding="max_length", max_length=128)

train = train.map(tokenize, batched=True)
val = val.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    BASE,
    num_labels=2,
    id2label={0: "NEGATIVE", 1: "POSITIVE"},
    label2id={"NEGATIVE": 0, "POSITIVE": 1},
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds),
            "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir="out",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    logging_steps=50,
    report_to="mlflow",
)

mlflow.set_experiment("sentiment-distilbert")
with mlflow.start_run():
    trainer = Trainer(model=model, args=args,
                      train_dataset=train, eval_dataset=val,
                      compute_metrics=compute_metrics)
    trainer.train()
    eval_res = trainer.evaluate()
    mlflow.log_metrics(eval_res)
    mlflow.transformers.log_model(
        transformers_model={"model": model, "tokenizer": tok},
        name="model",
        task="text-classification",
        registered_model_name="sentiment-distilbert",
        pip_requirements=[
            f"transformers=={transformers.__version__}",
            f"torch=={torch.__version__}",
            f"accelerate=={accelerate.__version__}",
        ],
    )