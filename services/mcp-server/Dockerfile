FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

# Install system dependencies including build tools for ML packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Create directory structure and fix permissions
RUN mkdir -p /app/app/tools /app/plugins /app/config /app/cache && \
    mkdir -p /home/hydrogen/.cache && \
    chmod 755 /home/hydrogen && \
    chmod 777 /home/hydrogen/.cache

# Copy and install requirements with better error handling
COPY requirements.txt /app/
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY ./app /app/app/

# Create __init__.py files
RUN touch /app/app/__init__.py /app/app/tools/__init__.py

# Create user and fix all permissions
RUN groupadd -r hydrogen && useradd -r -g hydrogen hydrogen && \
    chown -R hydrogen:hydrogen /app && \
    chown -R hydrogen:hydrogen /home/hydrogen

USER hydrogen

# Enhanced health check with longer startup time for ML models
HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]