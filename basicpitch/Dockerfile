FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    basic-pitch \
    numpy \
    librosa

# Create script to run basic-pitch
COPY run_basicpitch.py /app/

ENTRYPOINT ["python", "/app/run_basicpitch.py"]
