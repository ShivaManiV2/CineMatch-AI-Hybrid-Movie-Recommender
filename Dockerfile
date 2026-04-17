# Stage 1: Build Requirements
FROM python:3.9-slim AS builder

WORKDIR /app
COPY requirements.txt .

# Install build dependencies if needed
RUN apt-get update && apt-get install -y gcc g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt

# Stage 2: Final Image
FROM python:3.9-slim

WORKDIR /app

# Copy wheels from the builder stage
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .

# Install dependencies from wheels
RUN pip install --no-cache /wheels/*

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Command to run the FastApi app via Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
