"""Крок 8.4: змінює статус версії в SageMaker Model Registry.

Приклади (з кореня проєкту):
    python aws/set_approval.py 2 Approved "Метрики в межах похибки від v1, допускаємо до A/B-тесту"
    python aws/set_approval.py 1 Rejected "Програла A/B-тест"
"""
import sys

import boto3

from settings import REGION

GROUP = "sentiment-distilbert"
STATUSES = ("Approved", "Rejected", "PendingManualApproval")

# 1. Аргументи командного рядка: версія, статус, (необов'язково) причина.
if len(sys.argv) < 3 or sys.argv[2] not in STATUSES:
    raise SystemExit(f"Використання: python aws/set_approval.py <версія> <{'|'.join(STATUSES)}> \"причина\"")
version, status = sys.argv[1], sys.argv[2]
reason = sys.argv[3] if len(sys.argv) > 3 else ""

sm = boto3.Session(region_name=REGION).client("sagemaker")

# 2. Знаходимо в групі версію з потрібним номером.
packages = sm.list_model_packages(ModelPackageGroupName=GROUP)["ModelPackageSummaryList"]
package = next((p for p in packages if str(p["ModelPackageVersion"]) == version), None)
if package is None:
    raise SystemExit(f"У групі {GROUP} немає версії {version}")

# 3. Змінюємо статус і записуємо причину.
sm.update_model_package(
    ModelPackageArn=package["ModelPackageArn"],
    ModelApprovalStatus=status,
    ApprovalDescription=reason,
)
print(f"Версія {version}: {package['ModelApprovalStatus']} -> {status}")

# 4. Показуємо стан усього реєстру.
packages = sm.list_model_packages(ModelPackageGroupName=GROUP, SortBy="CreationTime",
                                  SortOrder="Ascending")["ModelPackageSummaryList"]
for p in packages:
    print(f"  v{p['ModelPackageVersion']}: {p['ModelApprovalStatus']}")