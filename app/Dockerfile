# Dockerfile (in /app)

FROM python:3.11-slim

# No buffering
ENV PYTHONUNBUFFERED=1

# App directory inside the container
WORKDIR /app

# Install system-level build tools (numpy/scikit-learn often need this)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY the app folder into the Docker context
COPY . /app

# Install dependencies from requirements-api.txt inside /app
RUN pip install --no-cache-dir -r requirements-api.txt

# Expose Cloud Run's required port
EXPOSE 8080

# Launch FastAPI via Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
