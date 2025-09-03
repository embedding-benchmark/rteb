#!/bin/bash

# Exit on error
set -e

# Activate the uv virtual environment
echo "Activating uv virtual environment..."
source /app/.venv/bin/activate
source /app/api_keys

# Run the application with any provided arguments
echo "Running RTEB with arguments: ${@:-none (using defaults)}"
python -m rteb "$@"
