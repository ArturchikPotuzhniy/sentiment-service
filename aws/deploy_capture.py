"""Крок 8.5: ендпоінт з data capture — кожен запит і відповідь записуються в S3.

Модель — найновіша Approved-версія з реєстру sentiment-distilbert (зараз v2).
Model → Endpoint config (+ DataCaptureConfig) → Endpoint.

Запуск з кореня проєкту: python aws/deploy_capture.py
Прибирання: python aws/delete_endpoint.py
"""
import time

import boto3

from settings import CAPTURE_S3_URI, ENDPOINT_NAME, INSTANCE_TYPE, REGION, ROLE_NAME

GROUP = "sentiment-distilbert"

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
sm = session.client("sagemaker")

role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
suffix = time.strftime("%Y%m%d-%H%M%S")

# 1. Найновіша схвалена версія з реєстру (пропускний пункт з 8.4).
packages = sm.list_model_packages(
    ModelPackageGroupName=GROUP,
    ModelApprovalStatus="Approved",
    SortBy="CreationTime",
    SortOrder="Descending",
)["ModelPackageSummaryList"]
if not packages:
    raise SystemExit(f"У групі {GROUP} немає Approved-версій — деплоїти нічого.")
version = packages[0]["ModelPackageVersion"]
package_arn = packages[0]["ModelPackageArn"]

# 2. Model: образ, архів і HF_TASK AWS бере з версії реєстру.
model_name = f"sentiment-distilbert-v{version}-{suffix}"
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=role_arn,
    PrimaryContainer={"ModelPackageName": package_arn},
)
print(f"Model: {model_name} (реєстр, версія {version})")

# 3. Endpoint config: та сама машина, що в 8.2, плюс DataCaptureConfig.
config_name = f"{model_name}-capture-config"
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName": "AllTraffic",
        "ModelName": model_name,
        "InstanceType": INSTANCE_TYPE,
        "InitialInstanceCount": 1,
    }],
    DataCaptureConfig={
        "EnableCapture": True,
        "InitialSamplingPercentage": 100,            # записувати кожен запит
        "DestinationS3Uri": CAPTURE_S3_URI,
        "CaptureOptions": [{"CaptureMode": "Input"}, {"CaptureMode": "Output"}],
        # JSON зберігати як текст, а не base64 — щоб файли можна було прочитати очима.
        "CaptureContentTypeHeader": {"JsonContentTypes": ["application/json"]},
    },
)
print("Endpoint config:", config_name)

# 4. Endpoint: з цього моменту AWS виділяє машину і починається оплата.
sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
print(f"Створюю ендпоінт {ENDPOINT_NAME} з data capture...")
start = time.time()
while True:
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"  {int(time.time() - start) // 60} хв: {status}")
    if status != "Creating":
        break
    time.sleep(30)

# 5. Результат: чи справді запис увімкнено і куди він іде.
endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
if status == "InService":
    capture = endpoint["DataCaptureConfig"]
    print(f"Data capture: {capture['CaptureStatus']}, "
          f"вибірка {capture['CurrentSamplingPercentage']}%")
    print(f"Файли з'являтимуться тут: {CAPTURE_S3_URI}/{ENDPOINT_NAME}/AllTraffic/рік/місяць/день/година/")
    print("Готово! Тепер надсилай трафік (крок 8.5.3).")
else:
    print("Не вдалося:", endpoint.get("FailureReason"))
    print(f"Логи: CloudWatch → Log groups → /aws/sagemaker/Endpoints/{ENDPOINT_NAME}")
print("Не забудь прибрати: python aws/delete_endpoint.py")