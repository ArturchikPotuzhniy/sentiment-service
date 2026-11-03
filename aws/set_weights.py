"""Крок 8.4: змінює розподіл трафіку між варіантами на ходу, без перестворення ендпоінта.

Приклади (з кореня проєкту):
    python aws/set_weights.py 1 9     # v1 : v2 = 1 : 9  ->  10% / 90%
    python aws/set_weights.py 1 1     # назад до 50 / 50
"""
import sys
import time

import boto3

from settings import ENDPOINT_NAME, REGION

# 1. Ваги з командного рядка.
if len(sys.argv) != 3:
    raise SystemExit("Використання: python aws/set_weights.py <вага v1> <вага v2>")
weights = {"v1": float(sys.argv[1]), "v2": float(sys.argv[2])}
if sum(weights.values()) == 0:
    raise SystemExit("Хоча б одна вага має бути більшою за 0")

sm = boto3.Session(region_name=REGION).client("sagemaker")

# 2. Нові ваги. Ендпоінт переходить в Updating, але продовжує відповідати.
sm.update_endpoint_weights_and_capacities(
    EndpointName=ENDPOINT_NAME,
    DesiredWeightsAndCapacities=[
        {"VariantName": name, "DesiredWeight": weight} for name, weight in weights.items()
    ],
)
print(f"Нові ваги: {weights}. Чекаю, поки застосуються...")

# 3. Чекаємо завершення оновлення.
while True:
    endpoint = sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
    if endpoint["EndpointStatus"] != "Updating":
        break
    time.sleep(10)

# 4. Поточні ваги і частки трафіку.
total = sum(v["CurrentWeight"] for v in endpoint["ProductionVariants"])
print("Статус:", endpoint["EndpointStatus"])
for v in endpoint["ProductionVariants"]:
    print(f"  {v['VariantName']}: вага {v['CurrentWeight']} -> {v['CurrentWeight'] / total:.0%} трафіку")