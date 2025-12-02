# Dockerfile (use this exact file)
FROM python:3.11-slim

WORKDIR /app
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Minimal OS deps for ffmpeg & opencv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      ffmpeg \
      libgl1 \
      wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install critical Python wheels first to avoid heavy builds
RUN pip install --upgrade pip wheel setuptools

# Install PyTorch CPU wheel (prebuilt) before requirements
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install small but critical libs explicitly to avoid dependency resolution cache issues
RUN pip install --no-cache-dir python-multipart pillow

# Copy requirements and install remaining deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy app code
COPY . /app

EXPOSE 8000
CMD ["uvicorn", "inference_server:app", "--host", "0.0.0.0", "--port", "8000"]
