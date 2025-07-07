# GhostNV GitHub Runners Setup Guide

## Runner Configuration

### nv-osmium (RTX 2060 - Turing)
**Purpose**: Legacy compatibility testing and validation
**Labels**: `[self-hosted, nv-osmium, rtx-2060]`

### nv-prometheus (RTX 3070 - Ampere)  
**Purpose**: Hybrid driver testing and NVENC validation
**Labels**: `[self-hosted, nv-prometheus, rtx-3070]`

### Home RTX 4090 (Ada Lovelace)
**Purpose**: Primary development target and gaming optimization
**Labels**: `[self-hosted, rtx-4090]`

## Runner Setup Instructions

### 1. GitHub Runner Installation

For both nv-osmium and nv-prometheus:

```bash
# Create a dedicated user for the runner
sudo useradd -m -s /bin/bash githubrunner
sudo usermod -a -G video,render githubrunner

# Switch to runner user
sudo su - githubrunner

# Download GitHub Actions runner
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64-2.311.0.tar.gz -L https://github.com/actions/runner/releases/download/v2.311.0/actions-runner-linux-x64-2.311.0.tar.gz
tar xzf ./actions-runner-linux-x64-2.311.0.tar.gz
```

### 2. Runner Configuration

#### nv-osmium (RTX 2060):
```bash
# Configure with specific labels
./config.sh --url https://github.com/ghostkellz/ghostnv --token [YOUR_TOKEN] \
  --name nv-osmium \
  --labels self-hosted,nv-osmium,rtx-2060,turing \
  --work _work
```

#### nv-prometheus (RTX 3070):
```bash
# Configure with specific labels  
./config.sh --url https://github.com/ghostkellz/ghostnv --token [YOUR_TOKEN] \
  --name nv-prometheus \
  --labels self-hosted,nv-prometheus,rtx-3070,ampere \
  --work _work
```

### 3. System Dependencies

Install required packages on both runners:

```bash
# Essential build tools
sudo apt update
sudo apt install -y \
  build-essential \
  linux-headers-$(uname -r) \
  curl \
  jq \
  git \
  lsb-release

# NVIDIA driver (if not already installed)
# Check: nvidia-smi
# If needed: sudo apt install nvidia-driver-545 nvidia-utils-545

# Docker (for container testing)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker githubrunner

# Reboot to ensure all modules loaded
sudo reboot
```

### 4. Hardware Validation

Test each runner's GPU access:

#### nv-osmium validation:
```bash
lspci | grep -i nvidia
# Expected: RTX 2060, device ID 1F15 or similar

nvidia-smi
# Expected: RTX 2060, 6GB VRAM, Turing architecture

ls -la /dev/nvidia*
# Expected: /dev/nvidia0, /dev/nvidiactl, /dev/nvidia-modeset
```

#### nv-prometheus validation:
```bash
lspci | grep -i nvidia  
# Expected: RTX 3070, device ID 2484 or similar

nvidia-smi
# Expected: RTX 3070, 8GB VRAM, Ampere architecture

# Test NVENC capabilities
ffmpeg -f lavfi -i testsrc2 -t 10 -c:v h264_nvenc -f null -
# Should work without errors
```

### 5. Runner Services

Set up systemd services for automatic startup:

#### nv-osmium service:
```bash
sudo tee /etc/systemd/system/github-runner-osmium.service > /dev/null <<EOF
[Unit]
Description=GitHub Actions Runner (nv-osmium)
After=network.target

[Service]
Type=simple
User=githubrunner
WorkingDirectory=/home/githubrunner/actions-runner
ExecStart=/home/githubrunner/actions-runner/run.sh
Restart=always
RestartSec=5
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable github-runner-osmium
sudo systemctl start github-runner-osmium
```

#### nv-prometheus service:
```bash
sudo tee /etc/systemd/system/github-runner-prometheus.service > /dev/null <<EOF
[Unit]
Description=GitHub Actions Runner (nv-prometheus)
After=network.target

[Service]
Type=simple
User=githubrunner
WorkingDirectory=/home/githubrunner/actions-runner
ExecStart=/home/githubrunner/actions-runner/run.sh
Restart=always
RestartSec=5
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable github-runner-prometheus
sudo systemctl start github-runner-prometheus
```

### 6. Testing Setup

Test runner connectivity:

```bash
# Check runner status
sudo systemctl status github-runner-osmium
sudo systemctl status github-runner-prometheus

# Check runner logs
sudo journalctl -u github-runner-osmium -f
sudo journalctl -u github-runner-prometheus -f

# Test GPU access as runner user
sudo su - githubrunner
nvidia-smi
lspci | grep -i nvidia
```

## CI Workflow Routing

### Test Distribution

#### nv-osmium (RTX 2060):
```yaml
test-turing:
  runs-on: [self-hosted, nv-osmium, rtx-2060]
```
- Legacy compatibility testing
- Turing architecture validation  
- OpenGL compatibility
- Basic GPU operations
- Performance baseline (450+ score)

#### nv-prometheus (RTX 3070):
```yaml
test-ampere:
  runs-on: [self-hosted, nv-prometheus, rtx-3070]
```
- Hybrid Zig/C driver testing
- NVENC H.264/H.265 encoding
- CUDA SM_86 compute capability
- RT cores 2nd gen testing
- Performance baseline (650+ score)

#### Home RTX 4090:
```yaml
test-ada:
  runs-on: [self-hosted, rtx-4090]
  if: github.event_name == 'schedule' || contains(github.event.head_commit.message, '[test-4090]')
```
- Pure Zig driver validation
- AV1 encoding (Ada exclusive)
- Digital vibrance hardware control
- RT cores 3rd gen + Tensor 4th gen
- Performance baseline (850+ score)

## Test Triggers

### Automatic Testing
- **Every push to main**: nv-osmium + nv-prometheus
- **Every PR**: nv-osmium + nv-prometheus  
- **Daily 2 AM UTC**: All runners including RTX 4090

### Manual Triggers
Include in commit messages:
- `[test-osmium]` - RTX 2060 specific tests
- `[test-prometheus]` - RTX 3070 specific tests
- `[test-4090]` - RTX 4090 home testing
- `[test-all]` - Full multi-GPU matrix

## Performance Monitoring

### Expected Baselines

**nv-osmium (RTX 2060)**:
- Memory bandwidth: ~448 GB/s
- Compute: ~450 GFLOPS  
- Overall score: 450+

**nv-prometheus (RTX 3070)**:
- Memory bandwidth: ~512 GB/s
- Compute: ~650 GFLOPS
- Overall score: 650+

**Home RTX 4090**:
- Memory bandwidth: ~1008 GB/s
- Compute: ~850 GFLOPS
- Overall score: 850+

### Alerts
- Performance drops >15% trigger investigation
- Test failures send notifications
- Hardware issues logged for analysis

## Troubleshooting

### Common Issues

#### Runner Not Connecting
```bash
# Check network connectivity
ping github.com

# Verify GitHub token permissions
# Token needs: repo, workflow, admin:org

# Check firewall
sudo ufw status
# Allow outbound HTTPS if needed
```

#### GPU Access Issues
```bash
# Check permissions
groups githubrunner
# Should include: video, render

# Check driver
nvidia-smi
# If fails: reinstall NVIDIA driver

# Check device files
ls -la /dev/nvidia*
sudo chmod 666 /dev/nvidia*  # if needed
```

#### Build Failures
```bash
# Check Zig installation
which zig
zig version

# Check disk space
df -h

# Clear old artifacts
rm -rf /home/githubrunner/actions-runner/_work/_temp/*
```

## Security Considerations

### Runner Isolation
- Dedicated `githubrunner` user
- Limited filesystem permissions
- No sudo access for runner process
- Regular security updates

### Network Security
- Outbound HTTPS only
- No inbound connections required
- VPN/firewall protection recommended
- Monitor runner activity logs

### Data Protection
- No sensitive data in repos
- Clear temp files after builds
- Encrypt runner storage if needed
- Regular backup of runner configs

## Maintenance

### Regular Tasks
- **Weekly**: Check runner health and logs
- **Monthly**: Update NVIDIA drivers
- **Quarterly**: Update runner software
- **As needed**: Clear disk space and temp files

### Updates
```bash
# Update runner software
cd /home/githubrunner/actions-runner
./svc.sh stop
curl -o update.tar.gz -L [new runner URL]
tar xzf update.tar.gz --overwrite
./svc.sh start

# Update NVIDIA drivers
sudo apt update && sudo apt upgrade nvidia-driver-*
sudo reboot
```

---

**Result**: Two dedicated GPU runners providing comprehensive multi-generation testing for GhostNV across Turing and Ampere architectures, with your RTX 4090 for Ada Lovelace development! 🚀