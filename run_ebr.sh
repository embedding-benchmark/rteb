#!/bin/bash

# Script to run memory profiling for EBR with Docker

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}EBR Setup${NC}"
echo "================================"

# Check if Docker is running
if ! sudo docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

echo ""
echo "Configuration:"
echo "- Memory Limit: 20GB"
echo ""

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
sudo docker-compose build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Run the profiling
echo ""
echo -e "${YELLOW}Starting memory profiling...${NC}"
echo "This may take a while depending on the dataset size."
echo "Profiling results will be saved to ./profiling_results/"
echo ""

# Run with docker-compose
sudo docker-compose up

# Check if profiling completed
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Completed successfully!${NC}"
    echo ""
else
    echo ""
    echo -e "${YELLOW}Container exited. ${NC}"
fi

# Cleanup
echo ""
echo "To remove the container, run: docker-compose down"
echo "To remove the image, run: docker-compose down --rmi all"
