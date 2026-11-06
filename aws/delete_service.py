"""Крок 8.6: видаляє публічний сервіс ECS Express разом із load balancer, HTTPS-сертифікатом
і security groups, які Express Mode створив для нього.

Ендпоінт SageMaker цей скрипт не чіпає — для нього python aws/delete_endpoint.py.
Запуск з кореня проєкту: python aws/delete_service.py
"""
import time

import boto3

from settings import ECS_SERVICE_NAME, REGION

session = boto3.Session(region_name=REGION)
ecs = session.client("ecs")
elbv2 = session.client("elbv2")
account = session.client("sts").get_caller_identity()["Account"]
service_arn = f"arn:aws:ecs:{REGION}:{account}:service/default/{ECS_SERVICE_NAME}"

# 1. Видалення сервісу: зупиняє контейнер і прибирає інфраструктуру, створену саме для нього.
ecs.delete_express_gateway_service(serviceArn=service_arn)
print(f"Видаляю {ECS_SERVICE_NAME}...")

# 2. Очікування: сервіс стає INACTIVE, а load balancer (погодинна оплата) зникає.
start = time.time()
while time.time() - start < 20 * 60:
    try:
        status = ecs.describe_express_gateway_service(serviceArn=service_arn)["service"]["status"]["statusCode"]
    except ecs.exceptions.ClientError:
        status = "INACTIVE"                         # сервісу вже немає зовсім
    balancers = [lb["LoadBalancerName"] for lb in elbv2.describe_load_balancers()["LoadBalancers"]]
    print(f"  {int(time.time() - start) // 60} хв: сервіс {status}, load balancers: {balancers or 'немає'}")
    if status == "INACTIVE" and not balancers:
        print("Готово: сервіс і load balancer видалено, оплата за них зупинена.")
        break
    time.sleep(30)
else:
    print("Load balancer ще не зник. Перевір: EC2 → Load Balancers (eu-north-1); якщо лишився — видали вручну.")
print("Ендпоінт SageMaker видаляється окремо: python aws/delete_endpoint.py")