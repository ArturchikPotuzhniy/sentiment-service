"""Крок 8.5: трафік в ендпоінт з data capture — звичайний (рецензії) або зсунутий (нейтральні твіти).

Ті самі тексти, що у фазі 7 (monitoring/send_traffic.py), але замість httpx.post
на localhost — invoke_endpoint. Кожен запит отримує мітку InferenceId
(<режим>-<час запуску>-<номер>); ендпоінт записує її в S3 разом із запитом,
і за нею analyze_capture.py відрізнить «норму» від «дрейфу».

Запуск з кореня проєкту:
  python aws/send_traffic.py normal
  python aws/send_traffic.py drift
"""
import argparse
import json
import statistics
import time

import boto3
from datasets import load_dataset

from settings import CAPTURE_S3_URI, ENDPOINT_NAME, REGION


def load_texts(mode: str, n: int) -> list[str]:
    # Без змін з фази 7: ті самі набори і той самий seed — результати можна порівняти.
    if mode == "normal":
        ds = load_dataset("cornell-movie-review-data/rotten_tomatoes", split="test")
    else:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="test")
        ds = ds.filter(lambda x: x["label"] == 1)  # 1 = neutral
    ds = ds.shuffle(seed=0).select(range(n))
    return list(ds["text"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["normal", "drift"])
    parser.add_argument("-n", type=int, default=200)
    args = parser.parse_args()

    texts = load_texts(args.mode, args.n)
    runtime = boto3.Session(region_name=REGION).client("sagemaker-runtime")

    # Мітка запуску: потрапить у кожен захоплений запис (eventMetadata.inferenceId).
    run_id = f"{args.mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    latencies = []
    positives = 0
    for i, text in enumerate(texts, 1):
        start = time.perf_counter()
        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=json.dumps({"inputs": text, "parameters": {"truncation": True}}),
            InferenceId=f"{run_id}-{i:04d}",
        )
        result = json.loads(response["Body"].read())[0]
        latencies.append((time.perf_counter() - start) * 1000)
        positives += result["label"] == "POSITIVE"
        if i % 50 == 0:
            print(f"надіслано {i}/{len(texts)}")

    print(f"готово: мітка запуску {run_id}")
    print(f"  POSITIVE {positives / len(texts):.0%}, "
          f"медіанна затримка {statistics.median(latencies):.0f} мс")
    print("Файли capture з'являться в S3 за кілька хвилин:")
    print(f"  aws s3 ls {CAPTURE_S3_URI}/{ENDPOINT_NAME}/ --recursive")


if __name__ == "__main__":
    main()