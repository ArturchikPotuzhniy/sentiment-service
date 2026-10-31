"""Крок 8.1: дістає модель з MLflow Model Registry і пакує її в model.tar.gz для SageMaker.

Запуск з кореня проєкту (там, де лежать mlflow.db і mlruns/):
    python aws/package_model.py
"""
import json
import shutil
import tarfile
from pathlib import Path

import mlflow
from transformers import pipeline

MODEL_NAME = "sentiment-distilbert"  # назва в Model Registry (фаза 5)
MODEL_VERSION = "1"                  # яку версію пакуємо: перевір у mlflow ui → Models

BUILD_DIR = Path("build")
MODEL_DIR = BUILD_DIR / "model"
ARCHIVE = BUILD_DIR / "model.tar.gz"

# 1. Завантажуємо модель і токенізатор з реєстру MLflow.
#    Трекінг-URI не задаємо: MLflow бере той самий, що й під час навчання
#    (mlflow.db у поточній папці), тому запускати треба з кореня проєкту.
print(f"Завантажую {MODEL_NAME} v{MODEL_VERSION} з MLflow...")
components = mlflow.transformers.load_model(
    f"models:/{MODEL_NAME}/{MODEL_VERSION}", return_type="components"
)
model = components["model"]
tokenizer = components["tokenizer"]

# 2. Зберігаємо у стандартному форматі Hugging Face в чисту папку build/model.
if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
MODEL_DIR.mkdir(parents=True)
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

# 2а. Сумісність з transformers 4.x у контейнері SageMaker.
#     transformers 5 зберігає токенізатор DistilBERT як BertTokenizer, а той додає
#     token_type_ids, яких DistilBERT не приймає. Явно вказуємо правильний клас і входи.
tok_cfg_path = MODEL_DIR / "tokenizer_config.json"
tok_cfg = json.loads(tok_cfg_path.read_text(encoding="utf-8"))
tok_cfg["tokenizer_class"] = "DistilBertTokenizer"
tok_cfg["model_input_names"] = ["input_ids", "attention_mask"]
tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False), encoding="utf-8")

# 3. Санітарна перевірка: модель з папки завантажується і повертає NEGATIVE/POSITIVE.
#    Зламану модель краще знайти зараз, ніж після 10 хвилин старту ендпоінта.
clf = pipeline("text-classification", model=str(MODEL_DIR))
assert "token_type_ids" not in clf.tokenizer("test"), "токенізатор досі видає token_type_ids"
for text in ["This is the best movie I have ever seen",
             "This was a terrible, boring waste of time"]:
    print(f"  {text!r} -> {clf(text)[0]}")

# 4. Пакуємо в model.tar.gz. Файли мають лежати в КОРЕНІ архіву (без папки model/),
#    бо SageMaker розпаковує архів прямо в /opt/ml/model.
with tarfile.open(ARCHIVE, "w:gz") as tar:
    for f in sorted(MODEL_DIR.iterdir()):
        tar.add(f, arcname=f.name)

print("Вміст архіву:")
with tarfile.open(ARCHIVE) as tar:
    for name in tar.getnames():
        print("  ", name)
print(f"Готово: {ARCHIVE} ({ARCHIVE.stat().st_size / 1e6:.0f} МБ)")