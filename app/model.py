"""Модель працює в SageMaker: цей модуль лише передає текст на ендпоінт і повертає відповідь."""
import json
import os

import boto3

ENDPOINT_NAME = os.getenv("SAGEMAKER_ENDPOINT", "sentiment-distilbert")
REGION = os.getenv("AWS_REGION", "eu-north-1")

# Клієнт створюється один раз при старті сервера — як раніше pipeline(...).
# Ключів тут немає: boto3 сам знайде права (локально — ваш профіль AWS CLI, на ECS — task role).
_runtime = boto3.client("sagemaker-runtime", region_name=REGION)


def predict(text: str) -> dict:
    response = _runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Body=json.dumps({"inputs": text, "parameters": {"truncation": True}}),
    )
    result = json.loads(response["Body"].read())[0]
    return {"label": result["label"], "score": float(result["score"])}
