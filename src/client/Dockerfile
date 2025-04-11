# Frontend Dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir nicegui requests

CMD ["python", "frontend.py"]
