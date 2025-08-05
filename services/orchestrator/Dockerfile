FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies with better error handling
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create directory structure first
RUN mkdir -p /app/app/agents /app/app/core /app/config /app/cache /app/logs /app/plugins

# Copy requirements first for better caching
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ./app /app/app/

# Create empty __init__.py files if they don't exist
RUN touch /app/app/__init__.py \
    /app/app/agents/__init__.py \
    /app/app/core/__init__.py

# Create user and set permissions
RUN groupadd -r hydrogen && useradd -r -g hydrogen hydrogen && \
    chown -R hydrogen:hydrogen /app

USER hydrogen

# Enhanced health check with better timeout
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Enhanced startup command with better error handling
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--log-level", "info"]