FROM python

# Set the working directory in the container to /app
WORKDIR /app
COPY requirements.txt .
RUN apt-get update && apt-get install -y poppler-utils
RUN pip install uv
RUN uv venv && uv pip install -r requirements.txt

# Copy the setup.py and project files into the container at /app
COPY papertuner ./papertuner

# Set environment variables (optional, but good practice for configuration)
# Example: Setting a default Ollama host if not provided during runtime
# Ollama on local host
ENV OLLAMA_HOST=http://localhost:11434

# Command to run when the container starts
# You can customize this to run specific parts of your papertuner project.
COPY example.py  .
COPY pyproject.toml  .
CMD ["uv", "run", "example.py"]
