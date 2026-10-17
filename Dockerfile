# syntax=docker/dockerfile:1.7
# Two final targets share the same application layer (docs/ARCHITECTURE.md §13.2):
#   runtime-cpu  python:3.11-slim + CPU torch   (api, CPU worker, model downloader)
#   runtime-gpu  CUDA 12 + cuDNN 9 + CUDA torch (GPU worker)

ARG PYTHON_IMAGE=python:3.11-slim-bookworm
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.8

FROM ${UV_IMAGE} AS uv

# ---------------------------------------------------------------- CPU runtime
FROM ${PYTHON_IMAGE} AS runtime-cpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/models/hf
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv venv /opt/venv --python /usr/local/bin/python3.11
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
# torch, torchaudio and torchcodec must come from the same PyTorch index (ABI-matched builds);
# everything else, including faster-whisper and pyannote.audio, comes from PyPI.
RUN uv pip install --python /opt/venv/bin/python --index-url ${TORCH_INDEX} \
        torch torchaudio torchcodec \
    && uv pip install --python /opt/venv/bin/python ".[ml]"
COPY scripts ./scripts
RUN useradd --system --uid 10001 --create-home app \
    && mkdir -p /data /models \
    && chown app:app /data /models
USER app
EXPOSE 8000
CMD ["transcriber-api"]

# ---------------------------------------------------------------- GPU runtime
FROM ${CUDA_IMAGE} AS runtime-gpu
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    XDG_CACHE_HOME=/tmp/cache \
    MPLCONFIGDIR=/tmp/matplotlib \
    HF_HOME=/models/hf
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=uv /uv /usr/local/bin/uv
RUN uv python install 3.11 && uv venv /opt/venv --python 3.11
WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
# torch, torchaudio and torchcodec must come from the same PyTorch index (ABI-matched builds);
# everything else, including faster-whisper and pyannote.audio, comes from PyPI.
RUN uv pip install --python /opt/venv/bin/python --index-url ${TORCH_INDEX} \
        torch torchaudio torchcodec \
    && uv pip install --python /opt/venv/bin/python ".[ml]"
COPY scripts ./scripts
RUN useradd --system --uid 10001 --create-home app \
    && mkdir -p /data /models \
    && chown app:app /data /models
USER app
CMD ["transcriber-worker"]
