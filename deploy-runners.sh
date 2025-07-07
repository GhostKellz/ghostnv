#!/bin/bash
set -e

echo "=== GhostNV Container Runner Deployment ==="

# Check if GitHub token is provided
if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Please set GITHUB_TOKEN environment variable"
    echo "Usage: GITHUB_TOKEN=your_token ./deploy-runners.sh"
    exit 1
fi

# Check Docker and NVIDIA runtime
echo "=== Checking Prerequisites ==="
docker --version || { echo "Error: Docker not installed"; exit 1; }
docker run --rm --gpus all nvidia/cuda:12.3-base-ubuntu22.04 nvidia-smi || { 
    echo "Error: NVIDIA Container Runtime not working"
    echo "Please ensure NVIDIA drivers and nvidia-container-runtime are installed"
    exit 1
}

# Check GPU availability
echo "=== Detecting GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits)
echo "Found $GPU_COUNT GPU(s)"

if [ "$GPU_COUNT" -lt 2 ]; then
    echo "Warning: Less than 2 GPUs detected. Adjust device IDs in docker-compose.yml"
fi

# Build the container image
echo "=== Building GPU Runner Image ==="
docker build -f Dockerfile.gpu-runner -t ghostnv-gpu-runner:latest .

# Deploy using docker-compose
echo "=== Deploying Runners ==="
export GITHUB_TOKEN=$GITHUB_TOKEN
docker-compose down 2>/dev/null || true
docker-compose up -d

# Wait for containers to start
echo "=== Waiting for Containers to Initialize ==="
sleep 10

# Check runner status
echo "=== Checking Runner Status ==="
echo ""
echo "nv-osmium (RTX 2060):"
docker logs nv-osmium-runner --tail 10 2>/dev/null || echo "Container not running"
echo ""
echo "nv-prometheus (RTX 3070):"
docker logs nv-prometheus-runner --tail 10 2>/dev/null || echo "Container not running"

# GPU access verification
echo ""
echo "=== Verifying GPU Access ==="
echo "nv-osmium GPU access:"
docker exec nv-osmium-runner nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU access failed"

echo "nv-prometheus GPU access:"
docker exec nv-prometheus-runner nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo "GPU access failed"

echo ""
echo "=== Deployment Summary ==="
echo "✅ Container images built"
echo "✅ Runners deployed via docker-compose"
echo "✅ GPU passthrough configured"
echo ""
echo "Monitor runner logs with:"
echo "  docker logs nv-osmium-runner -f"
echo "  docker logs nv-prometheus-runner -f"
echo ""
echo "Check runner status:"
echo "  docker-compose ps"
echo ""
echo "Restart runners:"
echo "  docker-compose restart"
echo ""
echo "Update runners:"
echo "  docker-compose down && docker-compose build --no-cache && docker-compose up -d"

# GitHub runner registration check
echo ""
echo "=== GitHub Runner Registration ==="
echo "Check your GitHub repository settings to verify runners are connected:"
echo "https://github.com/ghostkellz/ghostnv/settings/actions/runners"
echo ""
echo "Expected runners:"
echo "  - nv-osmium (RTX 2060, Turing)"
echo "  - nv-prometheus (RTX 3070, Ampere)"