FROM python:3.15.1-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas==3.1.0 scikit-learn==1.9.1 skops==0.16.0 fastapi==0.161.1 uvicorn==0.54.0

COPY app.py .
COPY artifacts/churn_pipeline.skops ./artifacts/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]