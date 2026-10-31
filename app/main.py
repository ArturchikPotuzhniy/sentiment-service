import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from app.model import predict

app = FastAPI(title="Sentiment API")

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "predictions.jsonl"
LOG_FILE.parent.mkdir(exist_ok=True)
_log_lock = threading.Lock()


def log_prediction(record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    print(line, flush=True)
    with _log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")


class Request(BaseModel):
    text: str


class Response(BaseModel):
    label: str
    score: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=Response)
def predict_endpoint(req: Request):
    start = time.perf_counter()
    result = predict(req.text)
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction({
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "text_len": len(req.text),
        "label": result["label"],
        "score": round(result["score"], 4),
        "latency_ms": round(latency_ms, 1),
    })
    return result
