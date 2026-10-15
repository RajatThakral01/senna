# Viral Clips Automator — app image (UI + CLI + background job worker).
# CPU only. Models, downloads and outputs are mounted from the host (docker-compose.yml).
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_ANALYTICS_ENABLED=False \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

# ffmpeg (libass, loudnorm, ebur128, sidechaincompress), Postgres client for the
# schema/migrations, fonts (Liberation Sans stands in for Arial), OpenCV runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg postgresql-client fontconfig fonts-liberation2 fonts-dejavu-core \
        libgl1 libglib2.0-0 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# JS runtime yt-dlp needs to unlock HD YouTube formats
COPY --from=denoland/deno:bin-2.5.6 /deno /usr/local/bin/deno

WORKDIR /app
COPY requirements-docker.txt .
# CPU-only PyTorch wheels (no CUDA download), then the exact pinned set
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements-docker.txt

COPY . .
RUN chmod +x docker/entrypoint.sh db/init_db.sh

EXPOSE 7862
ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["python", "ui.py"]
