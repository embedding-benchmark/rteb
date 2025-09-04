FROM python:3.11-slim

# Set environment variables for better memory management and caching
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies in a single layer to reduce image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libz-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager and add to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy only dependency files first to leverage Docker cache
COPY pyproject.toml uv.lock ./

# Copy application code
COPY rteb ./rteb
COPY results ./results

# Create output directory
RUN mkdir -p ./output

# Install Python dependencies in a single layer
RUN uv sync && \
    uv pip install . && \
    uv pip install openai voyageai cohere tiktoken vertexai google-genai gritlm FlagEmbedding sentence_transformers

# Copy entrypoint script and make it executable
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/docker-entrypoint.sh"]
