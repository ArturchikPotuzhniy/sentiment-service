"""Крок 8.4: метрики A/B-ендпоінта з CloudWatch, окремо для кожного варіанта.

Invocations  — скільки запитів обробив варіант (за кожну хвилину);
ModelLatency — скільки часу «думала» модель, без дороги по мережі.
Дані з'являються в CloudWatch із затримкою 1–3 хвилини.
Запуск з кореня проєкту: python aws/variant_metrics.py
"""
from datetime import datetime, timedelta, timezone

import boto3

from settings import ENDPOINT_NAME, REGION

VARIANTS = ["v1", "v2"]
MINUTES = 60  # за скільки останніх хвилин дивимося

cw = boto3.Session(region_name=REGION).client("cloudwatch")
end = datetime.now(timezone.utc)
start = end - timedelta(minutes=MINUTES)


def metric(name, variant, statistic):
    """Точки однієї метрики одного варіанта, по одній на хвилину, від старих до нових."""
    points = cw.get_metric_statistics(
        Namespace="AWS/SageMaker",
        MetricName=name,
        Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT_NAME},
                    {"Name": "VariantName", "Value": variant}],
        StartTime=start,
        EndTime=end,
        Period=60,
        Statistics=[statistic],
    )["Datapoints"]
    return sorted(points, key=lambda p: p["Timestamp"])


for variant in VARIANTS:
    calls = metric("Invocations", variant, "Sum")
    latency = metric("ModelLatency", variant, "Average")
    total = sum(p["Sum"] for p in calls)
    # ModelLatency у CloudWatch — у мікросекундах, тому / 1000 = мілісекунди.
    avg_ms = sum(p["Average"] for p in latency) / len(latency) / 1000 if latency else 0
    print(f"{variant}: запитів за {MINUTES} хв: {total:.0f}, середня ModelLatency ≈ {avg_ms:.0f} мс")
    for p in calls:
        print(f"    {p['Timestamp'].astimezone():%H:%M}  {p['Sum']:.0f}")