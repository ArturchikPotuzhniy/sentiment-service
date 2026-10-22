from fastapi import FastAPI
from pydantic import BaseModel
from app.model import predict

app = FastAPI(title="Sentiment API")

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
    return predict(req.text)
