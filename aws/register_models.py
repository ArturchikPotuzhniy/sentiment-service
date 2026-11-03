"""Крок 8.4: реєструє моделі v1 і v2 у SageMaker Model Registry.

Група (model package group) sentiment-distilbert — «назва моделі».
Версія (model package) = образ + model.tar.gz + метрики + статус схвалення.
  v1 — модель, що вже працювала на ендпоінті (8.2): одразу Approved.
  v2 — нова з training job (8.3): PendingManualApproval, рішення за людиною.
Метрики беремо з MLflow, щоб не переписувати числа руками.

Запуск з кореня проєкту (там mlflow.db): python aws/register_models.py
"""
import boto3
from mlflow import MlflowClient

from settings import BUCKET, IMAGE_URI, INSTANCE_TYPE, REGION

GROUP = "sentiment-distilbert"         # група в реєстрі AWS
MLFLOW_MODEL = "sentiment-distilbert"  # назва моделі в реєстрі MLflow
VERSIONS = {
    "1": {"status": "Approved", "trained_on": "local-phase5"},
    "2": {"status": "PendingManualApproval", "trained_on": "sagemaker-training-job"},
}

sm = boto3.Session(region_name=REGION).client("sagemaker")
mlflow_client = MlflowClient()

# 1. Група: створюємо, якщо її ще немає.
groups = sm.list_model_package_groups(NameContains=GROUP)["ModelPackageGroupSummaryList"]
if not any(g["ModelPackageGroupName"] == GROUP for g in groups):
    sm.create_model_package_group(
        ModelPackageGroupName=GROUP,
        ModelPackageGroupDescription="DistilBERT, тональність відгуків (sentiment-service)",
    )
    print("Створено групу", GROUP)

# 2. Запобіжник: якщо версії вже є — не реєструємо вдруге.
existing = sm.list_model_packages(ModelPackageGroupName=GROUP)["ModelPackageSummaryList"]
if existing:
    raise SystemExit(f"У групі {GROUP} вже є версій: {len(existing)}. Нічого не роблю.")

# 3. Реєструємо по черзі: v1 стане версією 1, v2 — версією 2.
for version, info in VERSIONS.items():
    # Метрики цієї версії — з її run у MLflow.
    mlflow_version = mlflow_client.get_model_version(MLFLOW_MODEL, version)
    metrics = mlflow_client.get_run(mlflow_version.run_id).data.metrics

    response = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=f"v{version}: {info['trained_on']}",
        InferenceSpecification={
            "Containers": [{
                "Image": IMAGE_URI,
                "ModelDataUrl": f"s3://{BUCKET}/models/sentiment-distilbert/v{version}/model.tar.gz",
                "Environment": {"HF_TASK": "text-classification"},
            }],
            "SupportedContentTypes": ["application/json"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": [INSTANCE_TYPE],
        },
        ModelApprovalStatus=info["status"],
        CustomerMetadataProperties={
            "eval_accuracy": f"{metrics['eval_accuracy']:.4f}",
            "eval_f1": f"{metrics['eval_f1']:.4f}",
            "trained_on": info["trained_on"],
            "mlflow_run_id": mlflow_version.run_id,
        },
    )
    short_name = response["ModelPackageArn"].split("/", 1)[1]  # без номера акаунта
    print(f"{short_name} | {info['status']} | accuracy {metrics['eval_accuracy']:.4f}")