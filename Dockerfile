FROM python:3.10-slim

# Install system dependencies
# ffmpeg: required for audio extraction and whisperx
# libgl1, libglib2.0-0: required for opencv
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir pytest

# Copy the rest of the application
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Default port for FastAPI (though CLI is also supported)
EXPOSE 8000

# Default command to run the CLI help
CMD ["python", "-m", "app.cli", "--help"]
