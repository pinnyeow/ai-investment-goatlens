# ── GOATlens Dockerfile ──
# Multi-stage build: install dependencies, then run the FastAPI app.
# The frontend is served as a static file from the same container.

FROM python:3.11-slim

# Prevents Python from writing .pyc files and enables real-time logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (layer caching: this only re-runs when requirements.txt changes)
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# The FastAPI app references "../frontend/index.html" from the backend/ directory,
# so we set the working directory to backend/
WORKDIR /app/backend

# Render sets the PORT env var; default to 8000 for local testing
ENV PORT=8000

# Start the server
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
