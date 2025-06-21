#!/bin/bash

# Exit on error
set -e

# Function to handle cleanup on exit
cleanup() {
    echo "Container is shutting down, ensuring profiling data is saved..."
    # Give memray time to write data
    sleep 2
}

# Activate the uv virtual environment
echo "Activating uv virtual environment..."
source /app/.venv/bin/activate
source /app/api_keys

# Run the application with memray profiling, teeing output to a log file and stdout
echo "Running with profiling..."
python -m ebr \
    --gpus 1 \
    --save_path /app/output \
    --data_path /app/data \
    --overwrite
