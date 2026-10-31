"""Видаляє ендпоінт разом з його Endpoint config і Model.

Запускай наприкінці КОЖНОЇ сесії: python aws/delete_endpoint.py
"""
import boto3
from botocore.exceptions import ClientError

from settings import ENDPOINT_NAME, REGION

sm = boto3.Session(region_name=REGION).client("sagemaker")

try:
    endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
except ClientError as e:
    if "Could not find endpoint" in str(e):
        print(f"Ендпоінта {ENDPOINT_NAME} немає — нічого не працює і не тарифікується.")
        raise SystemExit
    raise  # інша помилка (наприклад, немає прав) — показуємо як є

# Дізнаємося, який config і яку модель використовує ендпоінт, щоб видалити і їх.
config_name = endpoint["EndpointConfigName"]
config = sm.describe_endpoint_config(EndpointConfigName=config_name)
model_names = [v["ModelName"] for v in config["ProductionVariants"]]

sm.delete_endpoint(EndpointName=ENDPOINT_NAME)
print("Видалено endpoint:", ENDPOINT_NAME, "(оплата зупинена)")
sm.delete_endpoint_config(EndpointConfigName=config_name)
print("Видалено endpoint config:", config_name)
for name in model_names:
    sm.delete_model(ModelName=name)
    print("Видалено model:", name)