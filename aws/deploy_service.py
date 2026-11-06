"""Крок 8.6: публічний FastAPI на ECS Express Mode (варіант B).

Образ з ECR (тег = git-коміт) + три ролі з create_ecs_roles.py → ECS сам створює
контейнер на Fargate, load balancer і HTTPS-адресу виду https://<...>.ecs.eu-north-1.on.aws.
FastAPI всередині викликає ендпоінт SageMaker, тому ендпоінт має вже працювати
(python aws/deploy_capture.py).

Запуск з кореня проєкту: python aws/deploy_service.py <тег образу в ECR>, напр. cb9170a
Прибирання: python aws/delete_service.py
"""
import sys
import time
import urllib.request

import boto3

from settings import (ECR_REPOSITORY, ECS_EXECUTION_ROLE, ECS_INFRA_ROLE, ECS_SERVICE_NAME,
                      ECS_TASK_ROLE, ENDPOINT_NAME, REGION)

if len(sys.argv) != 2:
    raise SystemExit("Вкажи тег образу з ECR: python aws/deploy_service.py cb9170a")
tag = sys.argv[1]

session = boto3.Session(region_name=REGION)
ecs = session.client("ecs")
ecr = session.client("ecr")
sm = session.client("sagemaker")
account = session.client("sts").get_caller_identity()["Account"]


def role_arn(name):
    return f"arn:aws:iam::{account}:role/{name}"


# 0. Перевірки до будь-яких витрат: образ з таким тегом є в ECR, ендпоінт працює.
ecr.describe_images(repositoryName=ECR_REPOSITORY, imageIds=[{"imageTag": tag}])  # якщо немає — помилка ImageNotFound
try:
    endpoint_status = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)["EndpointStatus"]
except sm.exceptions.ClientError:
    endpoint_status = "немає"
if endpoint_status != "InService":
    raise SystemExit(f"Ендпоінт {ENDPOINT_NAME}: {endpoint_status}. Спершу: python aws/deploy_capture.py")
image = f"{account}.dkr.ecr.{REGION}.amazonaws.com/{ECR_REPOSITORY}:{tag}"
print("Образ:", image)

# 1. Кластер — «папка» для сервісів ECS. Безкоштовний; якщо вже є, create_cluster просто поверне його.
ecs.create_cluster(clusterName="default")

# 2. Сервіс: з цього моменту AWS створює load balancer і запускає контейнер — починається оплата.
service = ecs.create_express_gateway_service(
    serviceName=ECS_SERVICE_NAME,
    executionRoleArn=role_arn(ECS_EXECUTION_ROLE),
    infrastructureRoleArn=role_arn(ECS_INFRA_ROLE),
    taskRoleArn=role_arn(ECS_TASK_ROLE),
    primaryContainer={
        "image": image,
        "containerPort": 8000,                      # uvicorn у Dockerfile слухає 8000 (Express за замовчуванням чекає 80)
        "environment": [{"name": "SAGEMAKER_ENDPOINT", "value": ENDPOINT_NAME}],
    },
    healthCheckPath="/health",                      # за замовчуванням /ping, якого у FastAPI немає
    scalingTarget={"minTaskCount": 1, "maxTaskCount": 1},  # навчальний режим: рівно один контейнер
)["service"]
service_arn = service["serviceArn"]
print("Сервіс:", service_arn)


def public_url():
    """Публічна адреса з опису сервісу (з'являється не одразу)."""
    described = ecs.describe_express_gateway_service(serviceArn=service_arn)["service"]
    for config in described.get("activeConfigurations", []):
        for path in config.get("ingressPaths", []):
            if path["accessType"] == "PUBLIC":
                endpoint = path["endpoint"]
                url = endpoint if endpoint.startswith("http") else f"https://{endpoint}"
                return url.rstrip("/")
    return None


# 3. Очікування: сервіс готовий, коли публічна адреса відповідає на /health.
start = time.time()
url = None
while time.time() - start < 20 * 60:
    url = url or public_url()
    state = "адреси ще немає"
    if url:
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=10) as response:
                if response.status == 200:
                    break
        except Exception as error:              # DNS ще не готовий, 503 від load balancer тощо
            state = f"{url} ще не відповідає ({type(error).__name__})"
    print(f"  {int(time.time() - start) // 60} хв: {state}")
    time.sleep(30)
else:
    raise SystemExit("За 20 хв сервіс не запрацював. Дивись ECS → Clusters → default → sentiment-service → Tasks/Logs")

print(f"Готово за {int(time.time() - start) // 60} хв. Публічна адреса: {url}")
print(f"  Swagger UI:  {url}/docs")
print(f"  Перевірка:   Invoke-RestMethod -Method Post -Uri {url}/predict "
      "-ContentType 'application/json' -Body '{\"text\": \"I loved this film\"}'")
print("Не забудь прибрати: python aws/delete_service.py, потім python aws/delete_endpoint.py")