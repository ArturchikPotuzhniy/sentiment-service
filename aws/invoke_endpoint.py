"""Крок 8.2: надсилає тексти в ендпоінт і друкує відповіді.

Запуск: python aws/invoke_endpoint.py
або зі своїм текстом: python aws/invoke_endpoint.py "What a great movie"
"""
import json
import sys
import time

import boto3

from settings import ENDPOINT_NAME, REGION

# sagemaker-runtime — окремий клієнт для викликів моделі (а "sagemaker" — для керування).
runtime = boto3.Session(region_name=REGION).client("sagemaker-runtime")

texts = sys.argv[1:] or [
    "This is the best movie I have ever seen",
    "This was a terrible, boring waste of time",
    "I loved this film",
]

for text in texts:
    # Формат запиту контейнера Hugging Face: {"inputs": ..., "parameters": ...}.
    # truncation=True обрізає тексти довші за 512 токенів замість помилки.
    payload = {"inputs": text, "parameters": {"truncation": True}}
    start = time.perf_counter()
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    latency_ms = (time.perf_counter() - start) * 1000
    result = json.loads(response["Body"].read())
    print(f"{text!r} -> {result[0]}  ({latency_ms:.0f} мс)")