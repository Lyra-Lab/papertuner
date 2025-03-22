FROM nvidia/cuda:12.1.0-base-ubuntu22.04

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget build-essential python3.10 python3-pip python3.10-dev git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Miniconda setup
ENV CONDA_DIR /opt/conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR
ENV PATH $CONDA_DIR/bin:$PATH

# Create conda environment
RUN conda create --name unsloth_env python=3.10 && \
    echo "source activate unsloth_env" > ~/.bashrc
ENV PATH $CONDA_DIR/envs/unsloth_env/bin:$PATH

# Install PyTorch and Unsloth
RUN conda install -n unsloth_env -y pytorch-cuda=12.1 pytorch cudatoolkit xformers -c pytorch -c nvidia -c xformers && \
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" && \
    pip install --no-deps trl peft accelerate bitsandbytes autoawq

# Marimo setup
WORKDIR /app
COPY --link requirements.txt .
RUN pip install -r requirements.txt
COPY --link app.py .

# Security and runtime config
EXPOSE 8080
RUN useradd -m app_user && chown -R app_user /app
USER app_user

# Health check and startup
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8080/health || exit 1
CMD ["marimo", "edit", "marimo-examples", "--host", "0.0.0.0", "-p", "8080", "--include-code"]