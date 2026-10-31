"""Крок 8.3: запускає training job у SageMaker і чекає результату.

1. Пакує train/sm_job/ (sm_train.py + requirements.txt) у sourcedir.tar.gz і кладе в S3.
2. create_training_job: образ, код, канали даних, машина, куди класти модель.
3. Раз на 30 с друкує статус; наприкінці — метрики і шлях до model.tar.gz.

Запуск з кореня проєкту: python aws/train_job.py
"""
import tarfile
import time
from pathlib import Path

import boto3

from settings import (BUCKET, CODE_S3_PREFIX, DATA_S3_URI, REGION, ROLE_NAME,
                      TRAIN_IMAGE_URI, TRAIN_INSTANCE_TYPE, TRAIN_OUTPUT_S3_URI)

SOURCE_DIR = Path("train/sm_job")      # що пакуємо і відправляємо в контейнер
ENTRY_POINT = "sm_train.py"            # який файл контейнер запустить
# Гіперпараметри -> аргументи sm_train.py: --epochs 1 --learning-rate 5e-05 --batch-size 16
HYPERPARAMETERS = {"epochs": "1", "learning-rate": "5e-05", "batch-size": "16"}
METRICS = ["eval_accuracy", "eval_f1", "eval_loss", "train_loss"]

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
s3 = session.client("s3")
sm = session.client("sagemaker")

role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
job_name = "sentiment-distilbert-" + time.strftime("%Y%m%d-%H%M%S")

# 1. Код: train/sm_job/ -> build/sourcedir.tar.gz -> S3.
archive = Path("build/sourcedir.tar.gz")
archive.parent.mkdir(exist_ok=True)
with tarfile.open(archive, "w:gz") as tar:
    for name in [ENTRY_POINT, "requirements.txt"]:
        tar.add(SOURCE_DIR / name, arcname=name)   # файли в корінь архіву

code_key = f"{CODE_S3_PREFIX}/{job_name}/sourcedir.tar.gz"
s3.upload_file(str(archive), BUCKET, code_key)
code_uri = f"s3://{BUCKET}/{code_key}"
print("Код:", code_uri)


def channel(name):
    """Канал даних: усе з s3://.../data/rotten_tomatoes/<name>/ -> /opt/ml/input/data/<name>/."""
    return {
        "ChannelName": name,
        "DataSource": {"S3DataSource": {
            "S3DataType": "S3Prefix",
            "S3Uri": f"{DATA_S3_URI}/{name}/",
            "S3DataDistributionType": "FullyReplicated",
        }},
    }


# 2. Training job: з цього моменту AWS виділяє машину і починається оплата.
sm.create_training_job(
    TrainingJobName=job_name,
    RoleArn=role_arn,
    AlgorithmSpecification={
        "TrainingImage": TRAIN_IMAGE_URI,
        "TrainingInputMode": "File",
        # Шаблони, за якими SageMaker шукає метрики в логах (рядки eval_accuracy=0.8320).
        "MetricDefinitions": [{"Name": m, "Regex": f"{m}=([0-9\\.]+)"} for m in METRICS],
    },
    HyperParameters={
        **HYPERPARAMETERS,
        # Службові: де взяти код і що запустити.
        "sagemaker_program": ENTRY_POINT,
        "sagemaker_submit_directory": code_uri,
        "sagemaker_region": REGION,
    },
    InputDataConfig=[channel("train"), channel("validation")],
    OutputDataConfig={"S3OutputPath": TRAIN_OUTPUT_S3_URI},
    ResourceConfig={
        "InstanceType": TRAIN_INSTANCE_TYPE,
        "InstanceCount": 1,
        "VolumeSizeInGB": 30,
    },
    StoppingCondition={"MaxRuntimeInSeconds": 3600},  # запобіжник: максимум 1 година
)
print(f"Створено training job {job_name}")
print("Логи наживо (у другому терміналі, коли статус стане Training):")
print(f"  aws logs tail /aws/sagemaker/TrainingJobs --log-stream-name-prefix {job_name} --follow")

# 3. Чекаємо: друкуємо статус, коли він змінюється.
start = time.time()
last = None
while True:
    desc = sm.describe_training_job(TrainingJobName=job_name)
    status, secondary = desc["TrainingJobStatus"], desc["SecondaryStatus"]
    if (status, secondary) != last:
        print(f"  {int(time.time() - start) // 60} хв: {status} / {secondary}")
        last = (status, secondary)
    if status in ("Completed", "Failed", "Stopped"):
        break
    time.sleep(30)

# 4. Результат.
if status == "Completed":
    print("\nМетрики:")
    for m in desc.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']} = {m['Value']:.4f}")
    print("Оплачений час:", desc.get("BillableTimeInSeconds"), "с")
    print("Модель:", desc["ModelArtifacts"]["S3ModelArtifacts"])
else:
    print("Не вдалося:", desc.get("FailureReason"))
    print(f"Логи: CloudWatch → Log groups → /aws/sagemaker/TrainingJobs → {job_name}")