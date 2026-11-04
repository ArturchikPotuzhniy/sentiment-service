"""Крок 8.5: запускає перевірку дрейфу як processing job на машині AWS.

Ноутбук: кладе monitoring/sm_job/drift_check.py у S3 і заповнює «бланк» create_processing_job.
AWS: виділяє ml.m5.large, копіює capture і код із S3 у контейнер, запускає drift_check.py,
копіює report.json у S3 і вимикає машину. Ендпоінт для цього не потрібен.

Запуск з кореня проєкту:
  python aws/drift_job.py                          # останній normal-… проти останнього drift-…
  python aws/drift_job.py <базовий> <поточний>     # конкретні мітки запусків
"""
import json
import sys
import time

import boto3

from settings import (BUCKET, CAPTURE_S3_URI, ENDPOINT_NAME, MONITORING_S3_PREFIX,
                      PROCESSING_IMAGE_URI, PROCESSING_INSTANCE_TYPE, REGION, ROLE_NAME)

SCRIPT = "monitoring/sm_job/drift_check.py"          # на ноутбуці, відносно кореня проєкту
CODE_DIR = "/opt/ml/processing/input/code"           # у контейнері
CAPTURE_DIR = "/opt/ml/processing/input/capture"     # у контейнері
OUTPUT_DIR = "/opt/ml/processing/output"             # у контейнері

session = boto3.Session(region_name=REGION)
iam = session.client("iam")
sm = session.client("sagemaker")
s3 = session.client("s3")

role_arn = iam.get_role(RoleName=ROLE_NAME)["Role"]["Arn"]
job_name = f"sentiment-drift-{time.strftime('%Y%m%d-%H%M%S')}"

# 1. Код у S3: звідти SageMaker скопіює його в контейнер.
code_key = f"code/drift-check/{job_name}/drift_check.py"
s3.upload_file(SCRIPT, BUCKET, code_key)
print(f"Код: s3://{BUCKET}/{code_key}")

# 2. Аргументи для drift_check.py — як «python drift_check.py --input-dir ...» у командному рядку.
arguments = ["--input-dir", CAPTURE_DIR, "--output-dir", OUTPUT_DIR,
             "--cloudwatch", "--endpoint-name", ENDPOINT_NAME, "--region", REGION]
if len(sys.argv) == 3:
    arguments += ["--baseline", sys.argv[1], "--current", sys.argv[2]]

report_prefix = f"{MONITORING_S3_PREFIX}/reports/{job_name}"


def s3_input(name, s3_uri, local_path):
    """Один вхід job: що скопіювати з S3 і в яку папку контейнера."""
    return {
        "InputName": name,
        "S3Input": {
            "S3Uri": s3_uri,
            "LocalPath": local_path,
            "S3DataType": "S3Prefix",              # усе, що починається з цього префікса
            "S3InputMode": "File",                 # скопіювати файли до старту скрипта
            "S3DataDistributionType": "FullyReplicated",
        },
    }


# 3. Processing job: образ, машина, входи, вихід, роль, ліміт часу.
sm.create_processing_job(
    ProcessingJobName=job_name,
    RoleArn=role_arn,
    AppSpecification={
        "ImageUri": PROCESSING_IMAGE_URI,
        "ContainerEntrypoint": ["python3", f"{CODE_DIR}/drift_check.py"],
        "ContainerArguments": arguments,
    },
    ProcessingInputs=[
        s3_input("capture", f"{CAPTURE_S3_URI}/{ENDPOINT_NAME}/", CAPTURE_DIR),
        s3_input("code", f"s3://{BUCKET}/{code_key}", CODE_DIR),
    ],
    ProcessingOutputConfig={"Outputs": [{
        "OutputName": "report",
        "S3Output": {
            "S3Uri": f"s3://{BUCKET}/{report_prefix}",
            "LocalPath": OUTPUT_DIR,
            "S3UploadMode": "EndOfJob",            # скопіювати в S3 після завершення скрипта
        },
    }]},
    ProcessingResources={"ClusterConfig": {
        "InstanceCount": 1,
        "InstanceType": PROCESSING_INSTANCE_TYPE,
        "VolumeSizeInGB": 5,
    }},
    StoppingCondition={"MaxRuntimeInSeconds": 900},  # запобіжник: не довше 15 хв
)
print(f"Processing job: {job_name} ({PROCESSING_INSTANCE_TYPE})")

# 4. Очікування: машина вимкнеться сама, коли скрипт завершиться.
start = time.time()
while True:
    job = sm.describe_processing_job(ProcessingJobName=job_name)
    status = job["ProcessingJobStatus"]
    print(f"  {int(time.time() - start) // 60} хв: {status}")
    if status not in ("InProgress", "Stopping"):
        break
    time.sleep(30)

# 5. Результат: звіт із S3.
if status == "Completed":
    seconds = (job["ProcessingEndTime"] - job["ProcessingStartTime"]).total_seconds()
    print(f"Готово: обробка тривала {seconds:.0f} с. Звіт: s3://{BUCKET}/{report_prefix}/report.json")
    body = s3.get_object(Bucket=BUCKET, Key=f"{report_prefix}/report.json")["Body"].read()
    report = json.loads(body)
    print(f"базовий: {report['baseline_run']}, поточний: {report['current_run']}")
    for key in report["current"]:
        base_value, cur_value = report["baseline"][key], report["current"][key]
        print(f"  {key:<18}{base_value:>10.3f}{cur_value:>10.3f}" if key != "n"
              else f"  {key:<18}{base_value:>10}{cur_value:>10}")
    print("[!] Можливий дрейф: " + "; ".join(report["alerts"]) if report["alerts"]
          else "Суттєвих змін не виявлено.")
else:
    print("Не вдалося:", job.get("FailureReason"), job.get("ExitMessage", ""))
print(f"Логи: CloudWatch → Log groups → /aws/sagemaker/ProcessingJobs → {job_name}/...")