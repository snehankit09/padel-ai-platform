FROM python:3.11-slim

# ffmpeg is needed for video validation/processing (PRD: FFmpeg in Video Processing layer).
# fonts-dejavu-core provides the exact .ttf path ml/common/clip_extraction.py's
# drawtext overlay (Part 8d) references directly (_LABEL_FONT_FILE) — a
# fontconfig family-name lookup instead would depend on *some* matching
# font existing at runtime, which this slim base image doesn't guarantee
# on its own.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
