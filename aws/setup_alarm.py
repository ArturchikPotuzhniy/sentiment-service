"""Крок 8.5: аларм CloudWatch і лист на пошту, коли перевірка дрейфу знаходить проблему.

SNS-тема sentiment-drift-alerts ← підписка на вашу пошту (прийде лист з підтвердженням).
Аларм sentiment-drift стежить за метрикою drift_alerts, яку надсилає drift_check.py:
  drift_alerts ≥ 1 → стан ALARM → повідомлення в тему → лист;
  наступна перевірка без дрейфу (drift_alerts = 0) → стан OK → ще один лист.

Запуск з кореня проєкту: python aws/setup_alarm.py <ваша пошта>
Повторний запуск безпечний: тема, підписка й аларм не дублюються.
"""
import sys

import boto3

from settings import ENDPOINT_NAME, REGION

TOPIC_NAME = "sentiment-drift-alerts"
ALARM_NAME = "sentiment-drift"
NAMESPACE = "sentiment-service"  # той самий, що в drift_check.py

if len(sys.argv) != 2:
    raise SystemExit("Вкажи пошту: python aws/setup_alarm.py you@example.com")
email = sys.argv[1]

session = boto3.Session(region_name=REGION)
sns = session.client("sns")
cloudwatch = session.client("cloudwatch")

# 1. Тема SNS — «розсилка»: хто в неї публікує, того повідомлення отримують усі підписники.
topic_arn = sns.create_topic(Name=TOPIC_NAME)["TopicArn"]
print("SNS-тема:", topic_arn)

# 2. Підписка пошти на тему. Поки не натиснете посилання в листі, листів не буде.
subscription = sns.subscribe(TopicArn=topic_arn, Protocol="email", Endpoint=email)["SubscriptionArn"]
if subscription == "pending confirmation":
    print(f"Підписка: чекає підтвердження — відкрий лист від AWS Notifications на {email} "
          "і натисни Confirm subscription (перевір і «Спам»)")
else:
    print("Підписка: підтверджена")

# 3. Аларм: правило, яке CloudWatch сам перевіряє щохвилини.
cloudwatch.put_metric_alarm(
    AlarmName=ALARM_NAME,
    AlarmDescription="drift_check.py знайшов ознаки дрейфу (drift_alerts >= 1)",
    Namespace=NAMESPACE,
    MetricName="drift_alerts",
    Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT_NAME}],
    Statistic="Maximum",               # найбільше значення за період
    Period=300,                        # період — 5 хвилин
    EvaluationPeriods=1,               # достатньо одного «поганого» періоду
    Threshold=1,
    ComparisonOperator="GreaterThanOrEqualToThreshold",
    TreatMissingData="ignore",         # немає нових перевірок — стан не змінюється
    AlarmActions=[topic_arn],          # перехід в ALARM → лист
    OKActions=[topic_arn],             # повернення в OK → лист
)
state = cloudwatch.describe_alarms(AlarmNames=[ALARM_NAME])["MetricAlarms"][0]["StateValue"]
print(f"Аларм {ALARM_NAME}: стан {state}")
print("Далі: python aws/drift_job.py — за кілька хвилин після Completed має прийти лист ALARM.")