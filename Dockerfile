FROM python:3.12-slim

# git is needed by pip to install tvdatafeed from GitHub
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as UID 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e . \
    && mkdir -p data/cache .streamlit \
    && chown -R appuser:appuser /app

USER appuser
ENV HOME=/home/appuser

# Default port 7860 (Hugging Face Spaces); Cloud Run overrides $PORT to 8080.
ENV PORT=7860
CMD exec streamlit run app/streamlit_app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
