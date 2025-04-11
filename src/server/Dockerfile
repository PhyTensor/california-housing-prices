# Backend/Server Dockerfile
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Copy source code
COPY . .

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    joblib \
    numpy \
    pydantic \
    scikit-learn

# Expose port
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
