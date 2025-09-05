# Dockerfile for elmo_bert microservice
FROM python:3.10

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN apt-get update && apt-get install -y build-essential gcc && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt

COPY .env .env

# Copy source code
COPY . .

# Expose port (change if needed)
EXPOSE 8889

# Start FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8889"]
