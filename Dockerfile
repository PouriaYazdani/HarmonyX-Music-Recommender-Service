# ---- Builder stage: build wheels for faster/fuller install caching ----
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first for layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip && pip wheel --no-cache-dir --no-deps -r requirements.txt -w /wheels

# ---- Runtime stage ----
FROM python:3.10-slim

# Create non-root user
RUN useradd -m -u 10001 appuser

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Install deps from prebuilt wheels
COPY --from=builder /wheels /wheels
COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install --no-cache /wheels/* && rm -rf /wheels

# Copy code
COPY . /app

# Entrypoint for model/data bootstrap + app start
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown -R appuser:appuser /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MODEL_DIR=/app/models \
    DATA_DIR=/app/data \
    PORT=8000

USER appuser
EXPOSE 8000

CMD ["/entrypoint.sh"]
