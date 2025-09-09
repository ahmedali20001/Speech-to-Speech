FROM python:3.10-slim

# Install system dependencies with fixes for hash mismatch
RUN rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* /tmp/* && \
    apt-get clean && \
    echo 'Acquire::http::Pipeline-Depth "0";' > /etc/apt/apt.conf.d/99fixbadproxy && \
    echo 'Acquire::http::No-Cache "true";' >> /etc/apt/apt.conf.d/99fixbadproxy && \
    echo 'Acquire::BrokenProxy "true";' >> /etc/apt/apt.conf.d/99fixbadproxy && \
    apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Preload models to cache them in the image
COPY preload_models.py .
RUN python preload_models.py

# Copy source code
COPY server.py .
COPY audio/ ./audio/
COPY temp/ ./temp/
COPY logs/ ./logs/

# Expose ports
EXPOSE 8765 8080

# Command to run the server
CMD ["python", "server.py"]