FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY data/ ./data/

WORKDIR /app/api

EXPOSE 5000

ENV FLASK_ENV=production
ENV PYTHONPATH=/app

CMD ["flask", "run"]