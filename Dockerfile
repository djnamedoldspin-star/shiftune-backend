# 1) Base: a slim Python image
FROM python:3.11-slim

# 2) Install system libraries librosa needs
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# 3) Set working folder
WORKDIR /app

# 4) Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5) Copy the rest of your code (main.py etc.)
COPY . .

# 6) Start FastAPI with uvicorn
# Render sets $PORT, so we use that
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
