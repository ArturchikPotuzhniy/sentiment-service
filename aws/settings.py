"""Спільні налаштування для скриптів у папці aws/."""
REGION = "eu-north-1"
BUCKET = "sagemaker-sentiment-arturchik"
ROLE_NAME = "SageMakerExecutionRole"

MODEL_VERSION = "2"
MODEL_S3_URI = f"s3://{BUCKET}/models/sentiment-distilbert/v{MODEL_VERSION}/model.tar.gz"

# Готовий контейнер Hugging Face для інференсу на CPU.
# Лежить в ECR акаунта AWS 763104351884; у ньому transformers 4.51.3 і PyTorch 2.6.
IMAGE_URI = (
    "763104351884.dkr.ecr.eu-north-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.6.0-transformers4.51.3-cpu-py312-ubuntu22.04"
)

INSTANCE_TYPE = "ml.m5.large"          # 2 vCPU, 8 ГБ RAM
ENDPOINT_NAME = "sentiment-distilbert"  # постійна «адреса» ендпоінта
# Крок 8.3: дані для training job (канали train і validation).
DATA_S3_PREFIX = "data/rotten_tomatoes"
DATA_S3_URI = f"s3://{BUCKET}/{DATA_S3_PREFIX}"
# Крок 8.3: training job.
# Офіційний образ AWS для тренування на PyTorch (CPU): torch 2.8, Python 3.12.
TRAIN_IMAGE_URI = (
    "763104351884.dkr.ecr.eu-north-1.amazonaws.com/"
    "pytorch-training:2.8.0-cpu-py312"
)
TRAIN_INSTANCE_TYPE = "ml.m5.xlarge"  # 4 vCPU, 16 ГБ RAM
CODE_S3_PREFIX = "code/sentiment-distilbert"  # куди класти sourcedir.tar.gz
TRAIN_OUTPUT_S3_URI = f"s3://{BUCKET}/models/sentiment-distilbert/training-output"
# Крок 8.5: куди ендпоінт записує захоплені запити й відповіді (data capture).
CAPTURE_S3_PREFIX = "datacapture"
CAPTURE_S3_URI = f"s3://{BUCKET}/{CAPTURE_S3_PREFIX}"
# Крок 8.5: перевірка дрейфу як processing job.
# Стандартний образ AWS для processing job (scikit-learn, Python 3); нам з нього потрібні лише Python і boto3.
PROCESSING_IMAGE_URI = (
    "662702820516.dkr.ecr.eu-north-1.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)
PROCESSING_INSTANCE_TYPE = "ml.m5.large"   # 2 vCPU, 8 ГБ; квота 4 (крок 8.5.1)
MONITORING_S3_PREFIX = "monitoring"         # звіти: monitoring/reports/<назва job>/report.json
# Крок 8.6: FastAPI на ECS Express Mode (варіант B).
ECR_REPOSITORY = "sentiment-service"                       # образ: <акаунт>.dkr.ecr.<регіон>.amazonaws.com/sentiment-service:<тег>
ECS_SERVICE_NAME = "sentiment-service"
ECS_EXECUTION_ROLE = "ecsTaskExecutionRole"                 # ECS: образ з ECR, логи в CloudWatch
ECS_INFRA_ROLE = "ecsInfrastructureRoleForExpressServices"  # ECS: load balancer, security groups, автоскейлінг
ECS_TASK_ROLE = "sentiment-service-task-role"               # код FastAPI: лише InvokeEndpoint на наш ендпоінт