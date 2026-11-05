FROM python:3.11-slim

WORKDIR /code

# Лише сервіс: FastAPI + boto3. Модель працює в SageMaker, тому ні torch, ні transformers тут не потрібні.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]