#!/bin/bash
set -e

cd /home/runner/actions-runner

echo "=== GhostNV GPU Runner Startup ==="
echo "Runner Name: ${RUNNER_NAME}"
echo "Labels: ${RUNNER_LABELS}"
echo "Repository: ${GITHUB_REPO_URL}"

# Verify GPU access
echo "=== GPU Hardware Detection ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || echo "Warning: nvidia-smi failed"
lspci | grep -i nvidia || echo "Warning: No NVIDIA devices detected via lspci"

# Configure runner if not already configured
if [ ! -f ".runner" ]; then
    echo "=== Configuring GitHub Actions Runner ==="
    
    if [ -z "$GITHUB_TOKEN" ]; then
        echo "Error: GITHUB_TOKEN environment variable is required"
        exit 1
    fi
    
    if [ -z "$RUNNER_NAME" ]; then
        echo "Error: RUNNER_NAME environment variable is required"
        exit 1
    fi
    
    ./config.sh \
        --url "${GITHUB_REPO_URL}" \
        --token "${GITHUB_TOKEN}" \
        --name "${RUNNER_NAME}" \
        --labels "${RUNNER_LABELS}" \
        --work _work \
        --unattended \
        --replace
    
    echo "Runner configuration completed"
else
    echo "Runner already configured"
fi

# Health check
echo "=== Runner Health Check ==="
echo "Work directory: $(pwd)/_work"
echo "Runner user: $(whoami)"
echo "Container hostname: $(hostname)"

# Start the runner
echo "=== Starting GitHub Actions Runner: ${RUNNER_NAME} ==="
./run.sh