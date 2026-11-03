"""Крок 8.4: A/B-ендпоінт — дві версії з реєстру за однією адресою.

Model ×2 (з версій SageMaker Model Registry) → Endpoint config з двома варіантами → Endpoint.
Ваги 1 і 1: трафік ділиться 50/50.

Запуск з кореня проєкту: python aws/deploy_ab.py
Прибирання (обов'язково, дві машини!): python aws/delete_endpoint.py
"""
import time

import boto3

from settings import ENDPOINT_NAME, INSTANCE_TYPE, REGION, ROLE_NAME

GROUP = "sentiment-distilbert"
VARIANTS = {"v1": 1, "v2": 2}  # назва варіанта -> номер версії в реєстрі
WEIGHTS = {"v1": 1, "v2": 1}   # відносні ваги трафіку: 1 і 1 = 50/50

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
sm = session.client("sagemaker")

role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
suffix = time.strftime("%Y%m%d-%H%M%S")

# 1. Схвалені версії з реєстру: {номер версії: ARN}.
packages = sm.list_model_packages(
    ModelPackageGroupName=GROUP, ModelApprovalStatus="Approved",
)["ModelPackageSummaryList"]
approved = {p["ModelPackageVersion"]: p["ModelPackageArn"] for p in packages}

# 2. Model для кожного варіанта — з версії реєстру.
production_variants = []
for variant, version in VARIANTS.items():
    if version not in approved:
        raise SystemExit(f"Версія {version} не Approved — деплоїти її не можна.")
    model_name = f"sentiment-distilbert-{variant}-{suffix}"
    sm.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role_arn,
        PrimaryContainer={"ModelPackageName": approved[version]},
    )
    print(f"Model: {model_name}  (реєстр, версія {version})")
    production_variants.append({
        "VariantName": variant,
        "ModelName": model_name,
        "InstanceType": INSTANCE_TYPE,
        "InitialInstanceCount": 1,
        "InitialVariantWeight": WEIGHTS[variant],
    })

# 3. Один Endpoint config з двома варіантами.
config_name = f"sentiment-distilbert-ab-{suffix}-config"
sm.create_endpoint_config(EndpointConfigName=config_name, ProductionVariants=production_variants)
print("Endpoint config:", config_name)

# 4. Ендпоінт: з цього моменту оплачуються ДВІ машини.
sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
print(f"Створюю ендпоінт {ENDPOINT_NAME} з двома варіантами...")
start = time.time()
while True:
    status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
    print(f"  {int(time.time() - start) // 60} хв: {status}")
    if status != "Creating":
        break
    time.sleep(30)

# 5. Результат: які варіанти працюють і з якими вагами.
endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
if status == "InService":
    for v in endpoint["ProductionVariants"]:
        print(f"  {v['VariantName']}: вага {v['CurrentWeight']}, машин {v['CurrentInstanceCount']}")
    print("Готово! Тепер: python aws/send_ab_traffic.py")
else:
    print("Не вдалося:", endpoint.get("FailureReason"))
print("Не забудь прибрати: python aws/delete_endpoint.py")