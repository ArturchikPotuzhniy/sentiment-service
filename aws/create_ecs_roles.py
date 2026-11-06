"""Крок 8.6: три ролі IAM для FastAPI на ECS Express Mode.

ecsTaskExecutionRole                    — для ECS: завантажити образ з ECR, писати логи в CloudWatch
ecsInfrastructureRoleForExpressServices — для ECS: створити load balancer, security groups, автоскейлінг
sentiment-service-task-role             — для коду FastAPI в контейнері: лише sagemaker:InvokeEndpoint
                                          на ендпоінт sentiment-distilbert

Запуск з кореня проєкту: python aws/create_ecs_roles.py
Повторний запуск безпечний: наявні ролі не перестворюються. Ролі IAM безкоштовні.
"""
import json

import boto3

from settings import (ECS_EXECUTION_ROLE, ECS_INFRA_ROLE, ECS_TASK_ROLE, ENDPOINT_NAME,
                      REGION)

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
account = session.client("sts").get_caller_identity()["Account"]


def trust(service):
    """Trust policy: який сервіс AWS може «вдягнути» роль."""
    return json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Principal": {"Service": service}, "Action": "sts:AssumeRole"}],
    })


def create_role(name, service, description):
    try:
        iam.create_role(RoleName=name, AssumeRolePolicyDocument=trust(service), Description=description)
        print(f"Створено роль {name}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"Роль {name} вже є")


# 1. Execution role: її «вдягає» агент ECS, щоб підготувати контейнер до запуску.
create_role(ECS_EXECUTION_ROLE, "ecs-tasks.amazonaws.com",
            "ECS agent: pull images from ECR, write logs to CloudWatch")
iam.attach_role_policy(
    RoleName=ECS_EXECUTION_ROLE,
    PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy",
)

# 2. Infrastructure role: її «вдягає» сам ECS, коли створює, змінює чи видаляє сервіс.
create_role(ECS_INFRA_ROLE, "ecs.amazonaws.com",
            "ECS Express Mode: manage load balancer, security groups, auto scaling")
iam.attach_role_policy(
    RoleName=ECS_INFRA_ROLE,
    PolicyArn="arn:aws:iam::aws:policy/service-role/AmazonECSInfrastructureRoleforExpressGatewayServices",
)

# 3. Task role: права коду FastAPI — рівно одна дія на рівно один ендпоінт.
create_role(ECS_TASK_ROLE, "ecs-tasks.amazonaws.com",
            "sentiment-service FastAPI: invoke the sentiment-distilbert SageMaker endpoint")
endpoint_arn = f"arn:aws:sagemaker:{REGION}:{account}:endpoint/{ENDPOINT_NAME}"
iam.put_role_policy(
    RoleName=ECS_TASK_ROLE,
    PolicyName="invoke-sentiment-endpoint",
    PolicyDocument=json.dumps({
        "Version": "2012-10-17",
        "Statement": [{"Effect": "Allow", "Action": "sagemaker:InvokeEndpoint", "Resource": endpoint_arn}],
    }),
)
print(f"Task role може лише: sagemaker:InvokeEndpoint на {endpoint_arn}")