# Use a Python base image for development
FROM python:3.11-slim-buster

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt to the working directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install uv && uv venv
RUN uv pip install --no-cache-dir -r requirements.txt

# For marimo
EXPOSE 2718

CMD ["uv", "run", "marimo", "edit", "src/train.py"]
