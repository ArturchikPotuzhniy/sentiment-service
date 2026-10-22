from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

_classifier = pipeline("sentiment-analysis", model=MODEL_NAME)

def predict(text: str) -> dict:
    result = _classifier(text)[0]
    return {"label": result["label"], "score": float(result["score"])}
