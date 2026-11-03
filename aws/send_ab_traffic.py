"""Крок 8.4: надсилає рецензії з правильними мітками в A/B-ендпоінт і порівнює варіанти.

Режим split (справжній A/B): ендпоінт сам обирає варіант для кожного запиту за вагами.
Режим both (чесне порівняння): кожну рецензію надсилаємо в обидва варіанти (TargetVariant).

Запуск з кореня проєкту:
    python aws/send_ab_traffic.py                         # split, 200 рецензій
    python aws/send_ab_traffic.py --mode both --n 1066    # обидва варіанти, уся test-частина
Кожна відповідь записується в build/ab/<режим>-<час>.jsonl
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import boto3
from datasets import load_dataset

from settings import ENDPOINT_NAME, REGION

VARIANTS = ["v1", "v2"]

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["split", "both"], default="split")
parser.add_argument("--n", type=int, default=200, help="скільки рецензій, максимум 1066")
args = parser.parse_args()

# 1. Рецензії з test-частини: моделі їх не бачили ні в тренуванні, ні у валідації.
test = (load_dataset("cornell-movie-review-data/rotten_tomatoes")["test"]
        .shuffle(seed=42).select(range(args.n)))

runtime = boto3.Session(region_name=REGION).client("sagemaker-runtime")


def ask(text, target=None):
    """Один запит до ендпоінта. Повертає: варіант, передбачення 0/1, score, затримку в мс."""
    extra = {"TargetVariant": target} if target else {}
    start = time.perf_counter()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": text, "parameters": {"truncation": True}}),
        **extra,
    )
    latency_ms = (time.perf_counter() - start) * 1000
    result = json.loads(response["Body"].read())[0]
    pred = 1 if result["label"] == "POSITIVE" else 0
    return response["InvokedProductionVariant"], pred, result["score"], latency_ms


# 2. Надсилаємо рецензії і записуємо кожну відповідь у файл.
out_path = Path("build/ab") / f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
out_path.parent.mkdir(parents=True, exist_ok=True)
targets = VARIANTS if args.mode == "both" else [None]
records = []
with open(out_path, "w", encoding="utf-8") as f:
    for i, row in enumerate(test):
        for target in targets:
            variant, pred, score, latency = ask(row["text"], target)
            record = {"i": i, "variant": variant, "label": row["label"], "pred": pred,
                      "score": round(score, 4), "latency_ms": round(latency, 1)}
            records.append(record)
            f.write(json.dumps(record) + "\n")
        if (i + 1) % 50 == 0:
            print(f"  надіслано {i + 1}/{args.n}")

# 3. Підсумок по кожному варіанту.
by_variant = defaultdict(list)
for r in records:
    by_variant[r["variant"]].append(r)

print(f"\nРежим {args.mode}, рецензій {args.n}, файл: {out_path}")
for variant in sorted(by_variant):
    rs = by_variant[variant]
    accuracy = sum(r["pred"] == r["label"] for r in rs) / len(rs)
    latencies = sorted(r["latency_ms"] for r in rs)
    print(f"  {variant}: запитів {len(rs)} ({len(rs) / len(records):.0%}), "
          f"accuracy {accuracy:.3f}, медіанна затримка {latencies[len(latencies) // 2]:.0f} мс")

# 4. Лише для both: де моделі не погодились — хто мав рацію.
if args.mode == "both":
    answers = defaultdict(dict)
    for r in records:
        answers[r["i"]][r["variant"]] = r["pred"]
        answers[r["i"]]["label"] = r["label"]
    disagree = [a for a in answers.values() if a["v1"] != a["v2"]]
    v1_right = sum(a["v1"] == a["label"] for a in disagree)
    print(f"  Розбіжностей: {len(disagree)} з {args.n}; "
          f"права v1: {v1_right}, права v2: {len(disagree) - v1_right}")