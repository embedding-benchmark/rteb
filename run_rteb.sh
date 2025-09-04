#!/bin/bash

# Script to run RTEB with Docker

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}RTEB Setup${NC}"
echo "================================"

# Function to check if sudo is needed for Docker
check_docker_sudo() {
    # Try docker info without sudo
    if docker info > /dev/null 2>&1; then
        echo "Docker is accessible without sudo"
        return 1  # sudo not needed
    else
        echo "Docker requires sudo privileges"
        return 0  # sudo needed
    fi
}

# Check if Docker is running and determine if sudo is needed
if check_docker_sudo; then
    DOCKER_CMD="sudo docker"
    DOCKER_COMPOSE_CMD="sudo docker-compose"
else
    DOCKER_CMD="docker"
    DOCKER_COMPOSE_CMD="docker-compose"
fi

# Check if Docker is running
if ! $DOCKER_CMD info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Extract memory limit from docker-compose.yml
MEMORY_LIMIT=$(grep -A1 "x-memory-limit:" docker-compose.yml | grep "memory_limit" | awk '{print $3}')
if [ -z "$MEMORY_LIMIT" ]; then
    # Try another approach
    MEMORY_LIMIT=$(grep -A2 "limits:" docker-compose.yml | grep "memory:" | awk '{print $2}')
    if [ -z "$MEMORY_LIMIT" ]; then
        # Final fallback
        MEMORY_LIMIT="20G (default)"
    fi
fi

echo ""
echo "Configuration:"
echo "- Memory Limit: $MEMORY_LIMIT"
echo ""

# Build the Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
$DOCKER_COMPOSE_CMD build \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g)

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to build Docker image${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Docker image built successfully${NC}"

# Run the application
echo ""
echo -e "${YELLOW}Starting RTEB...${NC}"
echo "This may take a while depending on the dataset size."
echo ""

# Check if GPU is requested
if echo "$@" | grep -q "\-\-gpus"; then
    # GPU mode requested
    echo -e "${YELLOW}GPU mode detected${NC}"
    export NVIDIA_VISIBLE_DEVICES=all
else
    # CPU mode (default)
    echo -e "${YELLOW}CPU mode detected${NC}"
    export NVIDIA_VISIBLE_DEVICES=none
fi

# Run with docker-compose and pass any arguments to the container
echo -e "${YELLOW}Running with arguments: $@${NC}"
$DOCKER_COMPOSE_CMD run --rm rteb "$@"

# Check if execution completed
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
echo "To remove the container, run: ${DOCKER_COMPOSE_CMD} down"
echo "To remove the image, run: ${DOCKER_COMPOSE_CMD} down --rmi all"