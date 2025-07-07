# GhostNV Container-Based GPU Runners Setup

## Overview

Setting up GitHub Actions runners in containers with GPU passthrough for consistent, isolated testing environments across different NVIDIA GPU generations.

## Container Architecture

### nv-osmium (RTX 2060 - Turing)
```
Host Server
├── RTX 2060 GPU (passthrough)
├── Container: nv-osmium-runner
│   ├── GitHub Actions Runner
│   ├── NVIDIA Container Runtime
│   ├── Zig development environment
│   └── GhostNV testing tools
```

### nv-prometheus (RTX 3070 - Ampere)
```
Host Server  
├── RTX 3070 GPU (passthrough)
├── Container: nv-prometheus-runner
│   ├── GitHub Actions Runner
│   ├── NVIDIA Container Runtime
│   ├── Zig development environment
│   └── GhostNV testing tools
```

## Host Server Prerequisites

### 1. NVIDIA Driver Installation
```bash
# Install latest NVIDIA driver on host
sudo apt update
sudo apt install nvidia-driver-545 nvidia-utils-545

# Verify installation
nvidia-smi
lspci | grep -i nvidia
```

### 2. Docker with NVIDIA Container Runtime
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install nvidia-container-runtime

# Configure Docker daemon
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF

sudo systemctl restart docker

# Test GPU access in container
docker run --rm --gpus all nvidia/cuda:12.3-base-ubuntu22.04 nvidia-smi
```

## Container Image Creation

### Base Dockerfile for GitHub Actions Runner

```dockerfile
# Dockerfile.gpu-runner
FROM nvidia/cuda:12.3-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    jq \
    git \
    build-essential \
    linux-headers-generic \
    ca-certificates \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create runner user
RUN useradd -m -s /bin/bash runner && \
    usermod -aG sudo runner && \
    echo "runner ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install GitHub Actions runner
USER runner
WORKDIR /home/runner

RUN mkdir actions-runner && cd actions-runner && \
    curl -o actions-runner-linux-x64-2.311.0.tar.gz -L \
    https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz && \
    tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz && \
    rm actions-runner-linux-x64-2.311.0.tar.gz

# Create runner directories
RUN mkdir -p /home/runner/actions-runner/_work

# Set entrypoint
COPY entrypoint.sh /entrypoint.sh
USER root
RUN chmod +x /entrypoint.sh
USER runner

ENTRYPOINT ["/entrypoint.sh"]
```

### Runner Entrypoint Script

```bash
# entrypoint.sh
#!/bin/bash
set -e

cd /home/runner/actions-runner

# Configure runner if not already configured
if [ ! -f ".runner" ]; then
    echo "Configuring GitHub Actions runner..."
    ./config.sh \
        --url ${GITHUB_REPO_URL} \
        --token ${GITHUB_TOKEN} \
        --name ${RUNNER_NAME} \
        --labels ${RUNNER_LABELS} \
        --work _work \
        --unattended \
        --replace
fi

# Start the runner
echo "Starting GitHub Actions runner: ${RUNNER_NAME}"
./run.sh
```

## Container Deployment

### 1. Build Container Images

```bash
# Build base GPU runner image
docker build -f Dockerfile.gpu-runner -t ghostnv-gpu-runner:latest .
```

### 2. Deploy nv-osmium (RTX 2060)

```bash
# Create and start nv-osmium container
docker run -d \
    --name nv-osmium-runner \
    --restart unless-stopped \
    --gpus '"device=0"' \
    -e GITHUB_REPO_URL="https://github.com/ghostkellz/ghostnv" \
    -e GITHUB_TOKEN="YOUR_GITHUB_TOKEN" \
    -e RUNNER_NAME="nv-osmium" \
    -e RUNNER_LABELS="self-hosted,nv-osmium,rtx-2060,turing,linux" \
    ghostnv-gpu-runner:latest

# Verify GPU access
docker exec nv-osmium-runner nvidia-smi
docker exec nv-osmium-runner lspci | grep -i nvidia
```

### 3. Deploy nv-prometheus (RTX 3070)

```bash
# Create and start nv-prometheus container  
docker run -d \
    --name nv-prometheus-runner \
    --restart unless-stopped \
    --gpus '"device=1"' \
    -e GITHUB_REPO_URL="https://github.com/ghostkellz/ghostnv" \
    -e GITHUB_TOKEN="YOUR_GITHUB_TOKEN" \
    -e RUNNER_NAME="nv-prometheus" \
    -e RUNNER_LABELS="self-hosted,nv-prometheus,rtx-3070,ampere,linux" \
    ghostnv-gpu-runner:latest

# Verify GPU access
docker exec nv-prometheus-runner nvidia-smi
docker exec nv-prometheus-runner lspci | grep -i nvidia
```

## Docker Compose Configuration

For easier management, use docker-compose:

```yaml
# docker-compose.yml
version: '3.8'

services:
  nv-osmium:
    build:
      context: .
      dockerfile: Dockerfile.gpu-runner
    container_name: nv-osmium-runner
    restart: unless-stopped
    environment:
      - GITHUB_REPO_URL=https://github.com/ghostkellz/ghostnv
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - RUNNER_NAME=nv-osmium
      - RUNNER_LABELS=self-hosted,nv-osmium,rtx-2060,turing,linux
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']  # RTX 2060
              capabilities: [gpu]

  nv-prometheus:
    build:
      context: .
      dockerfile: Dockerfile.gpu-runner
    container_name: nv-prometheus-runner
    restart: unless-stopped
    environment:
      - GITHUB_REPO_URL=https://github.com/ghostkellz/ghostnv
      - GITHUB_TOKEN=${GITHUB_TOKEN}
      - RUNNER_NAME=nv-prometheus
      - RUNNER_LABELS=self-hosted,nv-prometheus,rtx-3070,ampere,linux
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']  # RTX 3070
              capabilities: [gpu]
```

Deploy with:
```bash
# Set GitHub token
export GITHUB_TOKEN="your_github_token_here"

# Deploy both runners
docker-compose up -d

# Monitor logs
docker-compose logs -f
```

## CI Workflow Updates

The CI workflows now target the containerized runners:

### Multi-GPU CI
```yaml
# RTX 2060 container
test-turing:
  runs-on: [self-hosted, nv-osmium, rtx-2060]

# RTX 3070 container  
test-ampere:
  runs-on: [self-hosted, nv-prometheus, rtx-3070]
```

### Main CI
```yaml
# Runs on any available container
build:
  runs-on: [self-hosted, linux]
```

## Container Benefits

### 1. **Isolation**
- Each runner in separate container
- No interference between test runs
- Clean environment for each build

### 2. **Consistency**
- Identical base environment
- Reproducible test results
- Easy runner recreation

### 3. **Security**
- Container-level isolation
- Limited host access
- Easy monitoring and logging

### 4. **Maintenance**
- Easy updates via image rebuilds
- Container-level backup/restore
- Simple scaling to more GPUs

## Monitoring and Management

### Container Health Checks
```bash
# Check runner status
docker ps
docker logs nv-osmium-runner
docker logs nv-prometheus-runner

# GPU utilization monitoring
docker exec nv-osmium-runner nvidia-smi -l 5
docker exec nv-prometheus-runner nvidia-smi -l 5

# Runner connectivity
docker exec nv-osmium-runner curl -s https://api.github.com
```

### Maintenance Commands
```bash
# Restart runners
docker restart nv-osmium-runner nv-prometheus-runner

# Update runners (rebuild and redeploy)
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Clean up old containers
docker system prune -f
```

## Troubleshooting

### GPU Not Accessible
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3-base nvidia-smi

# Check device permissions
ls -la /dev/nvidia*

# Restart Docker daemon
sudo systemctl restart docker
```

### Runner Not Connecting
```bash
# Check runner logs
docker logs nv-osmium-runner

# Verify GitHub token
echo $GITHUB_TOKEN

# Test network connectivity
docker exec nv-osmium-runner ping github.com
```

### Performance Issues
```bash
# Monitor container resources
docker stats

# Check GPU memory usage
docker exec nv-osmium-runner nvidia-smi

# Review host system resources
htop
nvidia-smi
```

## Security Considerations

### 1. **Token Management**
- Use GitHub tokens with minimal required permissions
- Rotate tokens regularly
- Store tokens securely (environment variables, secrets)

### 2. **Container Security**
- Run containers as non-root user
- Limit container capabilities
- Regular base image updates
- Monitor container activity

### 3. **Network Security**
- Firewall configuration
- Outbound-only GitHub connections
- VPN/private network if needed

---

**Result**: Two containerized GitHub Actions runners with dedicated GPU passthrough, providing isolated and consistent testing environments for RTX 2060 (Turing) and RTX 3070 (Ampere) architectures! 🐳🚀