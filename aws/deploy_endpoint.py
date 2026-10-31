"""Крок 8.2: створює SageMaker-ендпоінт з моделлю з S3.

Три виклики API: Model → Endpoint config → Endpoint.
Запуск з кореня проєкту: python aws/deploy_endpoint.py
"""
import time

import boto3

from settings import (ENDPOINT_NAME, IMAGE_URI, INSTANCE_TYPE, MODEL_S3_URI,
                      MODEL_VERSION, REGION, ROLE_NAME)

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
sm = session.client("sagemaker")

# ARN ролі беремо з IAM за назвою, щоб не вписувати номер акаунта в код.
role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]

# Імена Model і Endpoint config мають бути унікальними, тому додаємо час створення.
suffix = time.strftime("%Y%m%d-%H%M%S")
model_name = f"sentiment-distilbert-v{MODEL_VERSION}-{suffix}"
config_name = f"{model_name}-config"

# 1. Model: який контейнер запускати, де лежить модель, яку роль «вдягати».
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,
    PrimaryContainer={
        "Image": IMAGE_URI,
        "ModelDataUrl": MODEL_S3_URI,
        "Environment": {"HF_TASK": "text-classification"},
    },
)
print("Model:", model_name)

# 2. Endpoint config: на якій машині і скільки машин.
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": INSTANCE_TYPE,
        "InitialInstanceCount": 1,
    }],
)
print("Endpoint config:", config_name)

# 3. Endpoint: з цього моменту AWS виділяє машину і починається оплата.
sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
print(f"Створюю ендпоінт {ENDPOINT_NAME} (зазвичай 5–10 хвилин)...")

start = time.time()
while True:
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"  {int(time.time() - start) // 60} хв: {status}")
    if status != "Creating":
        break
    time.sleep(30)

if status == "InService":
    print("Готово! Тепер: python aws/invoke_endpoint.py")
else:
    reason = sm.describe_endpoint(EndpointName=ENDPOINT_NAME).get("FailureReason")
    print("Не вдалося:", reason)
    print(f"Логи: CloudWatch → Log groups → /aws/sagemaker/Endpoints/{ENDPOINT_NAME}")
    print("Не забудь прибрати залишки: python aws/delete_endpoint.py")