# Running EBR with Docker

This document explains how to run the EBR (Embedding Benchmark and Retrieval) application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- NVIDIA Docker runtime (for GPU support)

## Quick Start

1. Make sure Docker is running on your system.

2. Run the application with default settings:
   ```bash
   ./run_ebr.sh
   ```
   This will use the application's built-in defaults:
   - Data path: "data/"
   - Save path: "output/"
   - CPU mode (no GPUs)

3. Run with custom arguments:
   ```bash
   ./run_ebr.sh --gpus 2 --batch_size 32 --save_embds
   ```

## Available Arguments

All arguments supported by the EBR application can be passed directly to the Docker container. Here are some common ones:

- `--gpus <num>`: Number of GPUs to use (default: 0)
- `--cpus <num>`: Number of CPUs to use (default: 1)
- `--batch_size <num>`: Batch size for encoding (default: 16)
- `--data_path <path>`: Path to the dataset (default: /app/data)
- `--save_path <path>`: Path to save output (default: /app/output)
- `--save_embds`: Save embeddings
- `--load_embds`: Load pre-computed embeddings
- `--overwrite`: Overwrite existing results

For a complete list of arguments, run:
```bash
./run_ebr.sh --help
```

## Docker Configuration

The Docker setup includes:

1. A Docker image with all necessary dependencies
2. Volume mounts for data and output
3. GPU support for accelerated processing
4. Memory limits to prevent out-of-memory errors

## Customizing the Docker Environment

To modify the Docker environment:

1. Edit `docker-compose.yml` to change resource limits or volume mounts
2. Edit `Dockerfile` to modify the base image or installed dependencies
3. Edit `docker-entrypoint.sh` to change default arguments or startup behavior

After making changes, rebuild the Docker image:
```bash
sudo docker-compose build
```