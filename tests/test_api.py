from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict():
    r = client.post("/predict", json={"text": "I loved this film"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= body["score"] <= 1.0


def test_obvious_positive():
    r = client.post("/predict", json={"text": "This is the best movie I have ever seen"})
    assert r.json()["label"] == "POSITIVE"


def test_obvious_negative():
    r = client.post("/predict", json={"text": "This was a terrible, boring waste of time"})
    assert r.json()["label"] == "NEGATIVE"


def test_missing_text_returns_422():
    r = client.post("/predict", json={"message": "hi"})
    assert r.status_code == 422