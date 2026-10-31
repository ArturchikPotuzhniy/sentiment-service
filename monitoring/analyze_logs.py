"""Порівнює два логи передбачень (базовий і поточний) і підсвічує ознаки дрейфу."""
import json
import statistics
import sys

LOW_CONF = 0.7  # score нижче цього вважаємо невпевненим (для 2 класів мінімум — 0.5)


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(rows):
    scores = [r["score"] for r in rows]
    latencies = sorted(r["latency_ms"] for r in rows)
    return {
        "n": len(rows),
        "positive_share": sum(r["label"] == "POSITIVE" for r in rows) / len(rows),
        "mean_score": statistics.mean(scores),
        "low_conf_share": sum(s < LOW_CONF for s in scores) / len(rows),
        "median_text_len": statistics.median(r["text_len"] for r in rows),
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))],
    }


def fmt(v):
    return f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}"


def main():
    base = summarize(load(sys.argv[1]))
    cur = summarize(load(sys.argv[2]))

    print(f"{'метрика':<18}{'базовий':>12}{'поточний':>12}")
    for key in base:
        print(f"{key:<18}{fmt(base[key])}{fmt(cur[key])}")

    alerts = []
    if abs(cur["positive_share"] - base["positive_share"]) > 0.15:
        alerts.append("частка POSITIVE змінилась більш ніж на 15 п.п.")
    if cur["low_conf_share"] - base["low_conf_share"] > 0.10:
        alerts.append("невпевнених передбачень стало більше на 10+ п.п.")
    ratio = cur["median_text_len"] / base["median_text_len"]
    if ratio > 1.5 or ratio < 0.67:
        alerts.append(f"медіанна довжина тексту змінилась у {ratio:.2f} раза")

    print()
    if alerts:
        print("[!] Можливий дрейф:")
        for a in alerts:
            print("   -", a)
    else:
        print("Суттєвих змін не виявлено.")


if __name__ == "__main__":
    main()