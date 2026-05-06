# FunASR Triton Inference Server

Accelerating FunASR with NVIDIA Triton Inference Server and vLLM.

## Directory Structure

```
triton_server/
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── run_server.sh           # Server startup script
├── http_client.py          # HTTP client for testing
├── grpc_client.py          # gRPC client for testing
├── benchmark_client.sh     # Benchmark script
├── model_repo_funasr/      # Triton model repository
│   └── funasr/
│       ├── config.pbtxt    # Model configuration
│       └── 1/              # Model version
└── assets/
    └── zh.wav              # Sample audio for testing
```

## Environment Setup

### Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- Docker Compose v2.0+

### Install NVIDIA Container Toolkit

```bash
# Add the package repositories
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

### Verify GPU Access

```bash
docker run --rm --gpus all nvidia/cuda:12.0-base-ubuntu22.04 nvidia-smi
```

## Quick Start with Docker Compose

The fastest way to get started:

```bash
cd triton_server
docker compose up
```

This will:
1. Pull the pre-built Docker image
2. Start the Triton server with GPU support
3. Expose HTTP (8000), gRPC (8001), and Metrics (8002) ports

To run in detached mode:
```bash
docker compose up -d
```

To stop the service:
```bash
docker compose down
```

## Build Docker Image

To use the pre-built docker image:
```bash
docker pull soar97/triton-fun-asr:25.06
```


To build the image from scratch:

```bash
docker build -t funasr-triton:latest -f Dockerfile .
```

## Run with Docker

### Start a Container

```bash
docker run -it --name "funasr-triton-server" \
    --gpus all \
    --net host \
    --shm-size=2g \
    -v $(pwd)/model_repo_funasr:/models/funasr_repo \
    funasr-triton:latest
```

## Start Triton Server (Without Docker)

If running inside a container or directly on the host:

```bash
# Basic usage
./run_server.sh

# With custom options
./run_server.sh --http-port 8000 --grpc-port 8001 --metrics-port 8002 --gpu 0
```

### Server Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model-repo` | `./model_repo_funasr` | Model repository path |
| `--http-port` | `8000` | HTTP port |
| `--grpc-port` | `8001` | gRPC port |
| `--metrics-port` | `8002` | Metrics port |
| `--gpu` | `0` | GPU device ID |
| `--verbose` | - | Enable verbose logging |

## Testing the Server

### Check Server Health

```bash
# HTTP health check
curl -v http://localhost:8000/v2/health/ready

# Check model status
curl http://localhost:8000/v2/models/funasr
```

### HTTP Client Test

Send a single inference request using HTTP:

```bash
python3 http_client.py --audio assets/zh.wav --server localhost:8000 --model funasr
```

**Options:**
- `--audio`: Path to audio file (required)
- `--server`: Triton server URL (default: `localhost:8000`)
- `--model`: Model name (default: `funasr`)

### gRPC Client Test

Send a single inference request using gRPC:

```bash
python3 grpc_client.py --audio assets/zh.wav --server localhost:8001 --model funasr
```

**Options:**
- `--audio`: Path to audio file (required)
- `--server`: Triton server URL (default: `localhost:8001`)
- `--model`: Model name (default: `funasr`)

## Benchmark

### Run Benchmark with Triton Client

The benchmark script uses [Triton-ASR-Client](https://github.com/yuekaizhang/Triton-ASR-Client) to evaluate performance on standard datasets.

```bash
./benchmark_client.sh
```

This will:
1. Clone the Triton-ASR-Client repository (if not exists)
2. Run evaluation on the specified dataset
3. Compute CER (Character Error Rate)

### Custom Benchmark Configuration

Edit `benchmark_client.sh` to modify:
- `num_task`: Number of concurrent tasks (default: 32)
- `dataset_name`: HuggingFace dataset name
- `subset_name`: Dataset subset
- `split_name`: Dataset split (train/test/validation)

**Example configurations:**

```bash
# AISHELL-1 test set
dataset_name=yuekai/aishell
subset_name=test
split_name=test

# SpeechIO benchmark
dataset_name=yuekai/speechio
subset_name=SPEECHIO_ASR_ZH00007
split_name=test
```

### Manual Benchmark Command

```bash
python3 Triton-ASR-Client/client.py \
    --server-addr localhost \
    --model-name funasr \
    --num-tasks 32 \
    --log-dir ./log_funasr \
    --huggingface_dataset yuekai/aishell \
    --subset_name test \
    --split_name test \
    --enable_text_normalization \
    --compute-cer
```

### Results

**Benchmark Details:**
- **Dataset:** SPEECHIO 007 test set (approx. 1 hour of audio)
- **Hardware:** Single NVIDIA H20 GPU
- **Model:** Fun-ASR-Nano with vLLM backend

| Concurrency | CER | Processing Time | P50 Latency | Avg Latency | RTF |
|-------------|-----|-----------------|-------------|-------------|-----|
| 8 | 7.04% | 44.56s | 450.99ms | 458.17ms | 0.0126 |
| 16 | 7.00% | 27.96s | 533.36ms | 571.19ms | 0.0079 |
| 32 | 7.07% | 24.51s | 952.93ms | 1001.56ms | 0.0069 |

*Note: RTF (Real Time Factor) = processing_time / audio_duration. Lower is better.*

## Model Configuration

The model configuration is in `model_repo_funasr/funasr/config.pbtxt`. Key parameters:

- `max_batch_size`: Maximum batch size for dynamic batching
- `preferred_batch_size`: Preferred batch sizes for batching
- `max_queue_delay_microseconds`: Maximum time to wait for batching
