FROM python:3.11-slim

WORKDIR /code

# CPU-версія torch: образ на кілька ГБ менший
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Модель — до коду, щоб зміни в коді не змушували її перекачувати
RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

COPY app/ ./app/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]