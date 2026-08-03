FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NBA_WINPROB_API_HOST=0.0.0.0
ENV NBA_WINPROB_API_PORT=7860

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 7860

# Hugging Face supplies 7860; Render supplies PORT (usually 10000).
CMD ["sh", "-c", "nba-winprob serve --host 0.0.0.0 --port ${PORT:-7860}"]
