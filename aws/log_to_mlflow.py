"""Крок 8.3: записує training job з SageMaker у локальний MLflow.

1. Бере опис job з AWS: гіперпараметри, метрики, машину, шляхи в S3.
2. Створює run в експерименті sentiment-distilbert (поруч із запуском фази 5).
3. Реєструє версію моделі sentiment-distilbert з посиланням на model.tar.gz у S3.

Запуск з кореня проєкту (там лежить mlflow.db):
    python aws/log_to_mlflow.py                                       # останній успішний job
    python aws/log_to_mlflow.py sentiment-distilbert-20261030-192938  # конкретний job
"""
import json
import sys

import boto3
import mlflow
from mlflow import MlflowClient

from settings import IMAGE_URI, MODEL_S3_URI, MODEL_VERSION, REGION

EXPERIMENT = "sentiment-distilbert"        # той самий експеримент, що у фазі 5
REGISTERED_MODEL = "sentiment-distilbert"  # та сама назва в реєстрі, де вже є версія 1

sm = boto3.Session(region_name=REGION).client("sagemaker")

# 1. Який job записуємо: з аргументу або останній успішний.
if len(sys.argv) > 1:
    job_name = sys.argv[1]
else:
    jobs = sm.list_training_jobs(
        NameContains="sentiment-distilbert", StatusEquals="Completed",
        SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
    )
    job_name = jobs["TrainingJobSummaries"][0]["TrainingJobName"]

desc = sm.describe_training_job(TrainingJobName=job_name)
print("Job:", job_name, "|", desc["TrainingJobStatus"])

# 2. Запобіжник: нова версія в MLflow має отримати той самий номер, що й папка в S3 (v2).
client = MlflowClient()
versions = [int(v.version) for v in client.search_model_versions(f"name='{REGISTERED_MODEL}'")]
next_version = max(versions, default=0) + 1
if str(next_version) != MODEL_VERSION:
    raise SystemExit(
        f"У реєстрі вже є версії {sorted(versions)}, наступна буде {next_version}, "
        f"а в settings.py MODEL_VERSION = {MODEL_VERSION}. Можливо, цей job уже записано."
    )

# 3. Дістаємо потрібне з опису job.
hp = desc["HyperParameters"]
metrics = {m["MetricName"]: m["Value"] for m in desc.get("FinalMetricDataList", [])}
metrics["billable_time_s"] = desc.get("BillableTimeInSeconds", 0)
channels = {c["ChannelName"]: c["DataSource"]["S3DataSource"]["S3Uri"]
            for c in desc["InputDataConfig"]}

# 4. Run в MLflow.
mlflow.set_experiment(EXPERIMENT)
with mlflow.start_run(run_name=job_name) as run:
    mlflow.log_params({
        # Ті самі назви, що Trainer записав у фазі 5, щоб run-и порівнювалися в одній таблиці.
        "num_train_epochs": hp["epochs"],
        "learning_rate": hp["learning-rate"],
        "per_device_train_batch_size": hp["batch-size"],
        "instance_type": desc["ResourceConfig"]["InstanceType"],
        "training_image": desc["AlgorithmSpecification"]["TrainingImage"],
        "train_data": channels["train"],
        "validation_data": channels["validation"],
    })
    mlflow.log_metrics(metrics)
    mlflow.set_tags({
        "source": "sagemaker",
        "sagemaker_training_job": job_name,
        "code_uri": hp["sagemaker_submit_directory"],
        "mlflow.note.content": f"Тренування в SageMaker ({job_name}). Модель: {MODEL_S3_URI}",
    })
    # Повний опис job як файл серед артефактів run.
    # Без ARN: у них номер акаунта, а mlruns/ може потрапити в Git.
    record = {k: v for k, v in desc.items()
              if k not in ("RoleArn", "TrainingJobArn", "ResponseMetadata")}
    mlflow.log_dict(json.loads(json.dumps(record, default=str)), "sagemaker/training_job.json")

# 5. Версія в реєстрі: посилання на model.tar.gz у S3 + зв'язок із run.
mv = client.create_model_version(
    name=REGISTERED_MODEL,
    source=MODEL_S3_URI,
    run_id=run.info.run_id,
    tags={
        "source": "sagemaker",
        "serving_image": IMAGE_URI,
        "training_output": desc["ModelArtifacts"]["S3ModelArtifacts"],
    },
    description=f"Натреновано training job {job_name}. "
                f"Формат: model.tar.gz для ендпоінта SageMaker (deploy_endpoint.py).",
)
print("Run:", run.info.run_id)
print(f"Зареєстровано {REGISTERED_MODEL} версії {mv.version} -> {mv.source}")
print("Подивитися: mlflow ui -> http://127.0.0.1:5000")