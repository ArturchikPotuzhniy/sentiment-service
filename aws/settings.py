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