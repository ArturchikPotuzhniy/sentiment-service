"""Крок 8.5: аналог analyze_logs.py для data capture — читає захоплені запити з S3 і шукає дрейф.

Ендпоінт сам записав кожен запит і відповідь у S3 (deploy_capture.py),
send_traffic.py позначив їх мітками InferenceId (normal-… / drift-…).
Скрипт збирає записи, ділить їх на запуски за мітками і порівнює базовий запуск
з поточним тими самими метриками й порогами, що у фазі 7.

Запуск з кореня проєкту:
  python aws/analyze_capture.py                         # останній normal-… проти останнього drift-…
  python aws/analyze_capture.py <базовий> <поточний>    # конкретні мітки запусків
"""
import base64
import json
import statistics
import sys
from collections import defaultdict

import boto3

from settings import BUCKET, CAPTURE_S3_PREFIX, ENDPOINT_NAME, REGION

LOW_CONF = 0.7  # як у фазі 7: score нижче цього вважаємо невпевненим


def decode(part):
    """Поле data із запису capture → об'єкт Python. Зазвичай це JSON-текст, але може бути base64."""
    data = part["data"]
    if part["encoding"] == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return json.loads(data)


def read_capture():
    """Усі захоплені записи ендпоінта з S3, згруповані за міткою запуску."""
    s3 = boto3.Session(region_name=REGION).client("s3")
    prefix = f"{CAPTURE_S3_PREFIX}/{ENDPOINT_NAME}/"
    runs = defaultdict(list)
    skipped = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read().decode("utf-8")
            for line in body.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                inference_id = record["eventMetadata"].get("inferenceId")
                if inference_id is None:          # запит без мітки, напр. з invoke_endpoint.py
                    skipped += 1
                    continue
                request = decode(record["captureData"]["endpointInput"])
                answer = decode(record["captureData"]["endpointOutput"])[0]
                run_id = inference_id.rsplit("-", 1)[0]   # normal-20261105-140630-0001 → normal-20261105-140630
                runs[run_id].append({
                    "label": answer["label"],
                    "score": answer["score"],
                    "text_len": len(request["inputs"]),
                })
    return runs, skipped


def summarize(rows):
    # Як у фазі 7, але без p95_latency_ms: затримки в capture немає (вона в CloudWatch).
    scores = [r["score"] for r in rows]
    return {
        "n": len(rows),
        "positive_share": sum(r["label"] == "POSITIVE" for r in rows) / len(rows),
        "mean_score": statistics.mean(scores),
        "low_conf_share": sum(s < LOW_CONF for s in scores) / len(rows),
        "median_text_len": statistics.median(r["text_len"] for r in rows),
    }


def find_drift(base, cur):
    """Ті самі три правила й пороги, що в analyze_logs.py. Повертає список тривог."""
    alerts = []
    if abs(cur["positive_share"] - base["positive_share"]) > 0.15:
        alerts.append("частка POSITIVE змінилась більш ніж на 15 п.п.")
    if cur["low_conf_share"] - base["low_conf_share"] > 0.10:
        alerts.append("невпевнених передбачень стало більше на 10+ п.п.")
    ratio = cur["median_text_len"] / base["median_text_len"]
    if ratio > 1.5 or ratio < 0.67:
        alerts.append(f"медіанна довжина тексту змінилась у {ratio:.2f} раза")
    return alerts


def latest(runs, mode):
    """Найновіший запуск режиму mode: мітки містять час, тому сортування = хронологія."""
    names = sorted(name for name in runs if name.startswith(f"{mode}-"))
    if not names:
        raise SystemExit(f"У S3 немає запусків {mode}-… Спершу: python aws/send_traffic.py {mode}")
    return names[-1]


def fmt(v):
    return f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}"


def main():
    runs, skipped = read_capture()
    print("Запуски в S3:")
    for name in sorted(runs):
        print(f"  {name}: {len(runs[name])} записів")
    if skipped:
        print(f"  без мітки: {skipped} (пропущено)")

    if len(sys.argv) == 3:
        base_id, cur_id = sys.argv[1], sys.argv[2]
        for run_id in (base_id, cur_id):
            if run_id not in runs:
                raise SystemExit(f"Запуску {run_id} у S3 немає (див. список вище).")
    else:
        base_id, cur_id = latest(runs, "normal"), latest(runs, "drift")

    base = summarize(runs[base_id])
    cur = summarize(runs[cur_id])
    print(f"\nбазовий: {base_id}, поточний: {cur_id}")
    print(f"{'метрика':<18}{'базовий':>12}{'поточний':>12}")
    for key in base:
        print(f"{key:<18}{fmt(base[key])}{fmt(cur[key])}")

    alerts = find_drift(base, cur)
    print()
    if alerts:
        print("[!] Можливий дрейф:")
        for a in alerts:
            print("   -", a)
    else:
        print("Суттєвих змін не виявлено.")


if __name__ == "__main__":
    main()