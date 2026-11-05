"""Тести API без AWS: замість ендпоінта SageMaker — заглушки."""
import io
import json

from botocore.response import StreamingBody
from botocore.stub import Stubber
from fastapi.testclient import TestClient

from app import main, model

client = TestClient(main.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(monkeypatch):
    # main.py викликає predict(); підміняємо її готовою відповіддю — до AWS запит не йде.
    monkeypatch.setattr(main, "predict", lambda text: {"label": "POSITIVE", "score": 0.97})
    r = client.post("/predict", json={"text": "I loved this film"})
    assert r.status_code == 200
    assert r.json() == {"label": "POSITIVE", "score": 0.97}


def test_missing_text_returns_422():
    r = client.post("/predict", json={"message": "hi"})
    assert r.status_code == 422


def test_model_calls_endpoint():
    # Перевіряємо сам model.py: що саме він надсилає в SageMaker і як розбирає відповідь.
    body = json.dumps([{"label": "NEGATIVE", "score": 0.91}]).encode()
    with Stubber(model._runtime) as stub:
        stub.add_response(
            "invoke_endpoint",
            {"Body": StreamingBody(io.BytesIO(body), len(body)), "ContentType": "application/json"},
            {
                "EndpointName": "sentiment-distilbert",
                "ContentType": "application/json",
                "Body": json.dumps({"inputs": "boring", "parameters": {"truncation": True}}),
            },
        )
        assert model.predict("boring") == {"label": "NEGATIVE", "score": 0.91}