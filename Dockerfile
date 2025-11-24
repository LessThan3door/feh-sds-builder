FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy EVERYTHING from repo root into container
COPY . .

# Install all dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Uvicorn
EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
