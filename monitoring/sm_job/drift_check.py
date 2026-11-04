"""Крок 8.5: перевірка дрейфу всередині processing job (виконується на машині AWS).

Те саме, що aws/analyze_capture.py, але:
  - файли capture не завантажуються через boto3 — SageMaker сам скопіював їх із S3
    у папку --input-dir (ProcessingInputs у aws/drift_job.py);
  - результат — report.json у папці --output-dir, яку SageMaker після завершення
    job копіює в S3 (ProcessingOutputConfig);
  - з прапорцем --cloudwatch метрики поточного запуску йдуть у CloudWatch.

Локальна перевірка без AWS-машини (з кореня проєкту, після aws s3 sync у build/capture):
  python monitoring/sm_job/drift_check.py --input-dir build/capture --output-dir build/drift_report
"""
import argparse
import base64
import json
import statistics
from collections import defaultdict
from pathlib import Path

LOW_CONF = 0.7  # як у фазі 7: score нижче цього вважаємо невпевненим
NAMESPACE = "sentiment-service"  # «папка» для наших метрик у CloudWatch


def decode(part):
    """Поле data із запису capture → об'єкт Python. Зазвичай це JSON-текст, але може бути base64."""
    data = part["data"]
    if part["encoding"] == "BASE64":
        data = base64.b64decode(data).decode("utf-8")
    return json.loads(data)


def read_capture(input_dir):
    """Усі файли *.jsonl у папці (з підпапками за годинами), згруповані за міткою запуску."""
    runs = defaultdict(list)
    skipped = 0
    files = sorted(Path(input_dir).rglob("*.jsonl"))
    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            inference_id = record["eventMetadata"].get("inferenceId")
            if inference_id is None:
                skipped += 1
                continue
            request = decode(record["captureData"]["endpointInput"])
            answer = decode(record["captureData"]["endpointOutput"])[0]
            runs[inference_id.rsplit("-", 1)[0]].append({
                "label": answer["label"],
                "score": answer["score"],
                "text_len": len(request["inputs"]),
            })
    return runs, skipped, len(files)


def summarize(rows):
    scores = [r["score"] for r in rows]
    return {
        "n": len(rows),
        "positive_share": sum(r["label"] == "POSITIVE" for r in rows) / len(rows),
        "mean_score": statistics.mean(scores),
        "low_conf_share": sum(s < LOW_CONF for s in scores) / len(rows),
        "median_text_len": statistics.median(r["text_len"] for r in rows),
    }


def find_drift(base, cur):
    alerts = []
    if abs(cur["positive_share"] - base["positive_share"]) > 0.15:
        alerts.append("частка POSITIVE змінилась більш ніж на 15 п.п.")
    if cur["low_conf_share"] - base["low_conf_share"] > 0.10:
        alerts.append("невпевнених передбачень стало більше на 10+ п.п.")
    ratio = cur["median_text_len"] / base["median_text_len"]
    if ratio > 1.5 or ratio < 0.67:
        alerts.append(f"медіанна довжина тексту змінилась у {ratio:.2f} раза")
    return alerts


def latest(runs, mode):
    names = sorted(name for name in runs if name.startswith(f"{mode}-"))
    if not names:
        raise SystemExit(f"Немає запусків {mode}-… у вхідних даних")
    return names[-1]


def publish(metrics, endpoint_name, region):
    """Метрики поточного запуску → CloudWatch (put_metric_data від імені ролі job)."""
    import boto3  # лише тут: для локальної перевірки без --cloudwatch boto3 не потрібен
    cloudwatch = boto3.client("cloudwatch", region_name=region)
    cloudwatch.put_metric_data(
        Namespace=NAMESPACE,
        MetricData=[
            {
                "MetricName": name,
                "Dimensions": [{"Name": "EndpointName", "Value": endpoint_name}],
                "Value": float(value),
            }
            for name, value in metrics.items()
        ],
    )
    print(f"CloudWatch: {len(metrics)} метрик у {NAMESPACE} (EndpointName={endpoint_name})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="/opt/ml/processing/input/capture")
    parser.add_argument("--output-dir", default="/opt/ml/processing/output")
    parser.add_argument("--baseline", help="мітка базового запуску; за замовчуванням останній normal-…")
    parser.add_argument("--current", help="мітка поточного запуску; за замовчуванням останній drift-…")
    parser.add_argument("--cloudwatch", action="store_true", help="надіслати метрики в CloudWatch")
    parser.add_argument("--endpoint-name", default="sentiment-distilbert")
    parser.add_argument("--region", default="eu-north-1")
    args = parser.parse_args()

    # 1. Дані: файли, які SageMaker скопіював із S3 у контейнер.
    runs, skipped, n_files = read_capture(args.input_dir)
    print(f"Файлів: {n_files}; запуски: " + ", ".join(f"{k} ({len(v)})" for k, v in sorted(runs.items())))
    if skipped:
        print(f"Без мітки: {skipped} (пропущено)")

    # 2. Порівняння — ті самі метрики й правила, що у фазі 7 і в analyze_capture.py.
    base_id = args.baseline or latest(runs, "normal")
    cur_id = args.current or latest(runs, "drift")
    base, cur = summarize(runs[base_id]), summarize(runs[cur_id])
    alerts = find_drift(base, cur)

    # 3. Звіт: усе, що SageMaker покладе в S3 після завершення job.
    report = {
        "baseline_run": base_id,
        "current_run": cur_id,
        "baseline": base,
        "current": cur,
        "alerts": alerts,
    }
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    # 4. Метрики в CloudWatch: на них у 8.5.6 повісимо аларм.
    if args.cloudwatch:
        metrics = {key: value for key, value in cur.items() if key != "n"}
        metrics["drift_alerts"] = len(alerts)
        publish(metrics, args.endpoint_name, args.region)


if __name__ == "__main__":
    main()