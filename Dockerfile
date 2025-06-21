FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libz-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Set uv to PATH for all subsequent commands (for root user)
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files (as root initially)
COPY pyproject.toml uv.lock ./
COPY ebr ./ebr
COPY data ./data
COPY results ./results
RUN mkdir ./output
# Install Python dependencies (as root)
RUN uv sync

# Install memray for profiling (as root)
RUN uv pip install .
RUN uv pip install openai voyageai cohere tiktoken vertexai google-genai gritlm FlagEmbedding sentence_transformers

# Ensure uv is in PATH for the appuser
ENV PATH="/root/.local/bin:${PATH}"

# Set environment variables for better memory management
ENV PYTHONUNBUFFERED=1

# Copy entrypoint script and ebr runner (as appuser)
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
