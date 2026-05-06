# Fun-ASR vLLM 接口文档

## 1. 项目概述

Fun-ASR vLLM 是一个基于 vLLM 加速的语音识别服务，通过 NVIDIA Triton Inference Server 提供高性能的语音转文本 API。该服务支持中文、英文等多种语言的语音识别，具有低延迟、高吞吐量的特点。

**核心特性：**
- 基于 vLLM 加速的高性能推理
- 支持批量推理，提高吞吐量
- 多语言支持
- 生产级部署方案

## 2. 服务部署

### 2.1 环境要求

- NVIDIA GPU 与 CUDA 支持
- Docker 与 NVIDIA Container Toolkit
- Docker Compose v2.0+

### 2.2 快速部署（推荐）

使用 Docker Compose 快速部署服务：

```bash
cd triton_server
docker compose up -d
```

### 2.3 服务状态检查

服务启动后，可通过以下命令检查服务健康状态：

```bash
# HTTP 健康检查
curl -v http://localhost:8000/v2/health/ready

# 模型状态检查
curl http://localhost:8000/v2/models/funasr
```

### 2.4 服务端口

| 端口 | 用途 | 协议 |
|------|------|------|
| 8000 | HTTP 接口 | HTTP |
| 8001 | gRPC 接口 | gRPC |
| 8002 | Metrics 接口 | HTTP |

## 3. API 接口详情

### 3.1 HTTP API

#### 3.1.1 语音识别接口

**端点：** `http://<server-address>:8000/v2/models/funasr/infer`

**方法：** POST

**请求格式：** JSON

**请求体示例：**

```json
{
  "inputs": [
    {
      "name": "WAV",
      "shape": [1, 16000],
      "datatype": "FP32",
      "data": [0.0, 0.1, 0.2, ...]  // 音频数据
    },
    {
      "name": "WAV_LENS",
      "shape": [1, 1],
      "datatype": "INT32",
      "data": [16000]  // 音频长度
    }
  ],
  "outputs": [
    {
      "name": "TRANSCRIPTS"
    }
  ]
}
```

**响应格式：**

```json
{
  "model_name": "funasr",
  "model_version": "1",
  "outputs": [
    {
      "name": "TRANSCRIPTS",
      "shape": [1, 1],
      "datatype": "BYTES",
      "data": ["这是一段语音识别结果"]
    }
  ]
}
```

#### 3.1.2 Python 客户端示例

```python
import numpy as np
import tritonclient.http as httpclient
import librosa

# 加载音频
def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio.astype(np.float32)

# 初始化客户端
client = httpclient.InferenceServerClient(url="localhost:8000")

# 准备音频数据
audio = load_audio("path/to/audio.wav")

# 准备输入
wav_input = httpclient.InferInput("WAV", [1, len(audio)], "FP32")
wav_input.set_data_from_numpy(audio.reshape(1, -1))

wav_lens_input = httpclient.InferInput("WAV_LENS", [1, 1], "INT32")
wav_lens_input.set_data_from_numpy(np.array([[len(audio)]], dtype=np.int32))

# 准备输出
transcript_output = httpclient.InferRequestedOutput("TRANSCRIPTS")

# 发送请求
response = client.infer(
    model_name="funasr",
    inputs=[wav_input, wav_lens_input],
    outputs=[transcript_output]
)

# 处理响应
transcript = response.as_numpy("TRANSCRIPTS")[0][0]
if isinstance(transcript, bytes):
    transcript = transcript.decode("utf-8")

print("识别结果:", transcript)
```

### 3.2 gRPC API

#### 3.2.1 语音识别接口

**服务：** `inference.GRPCInferenceService`

**方法：** `ModelInfer`

**请求参数：**
- `model_name`: "funasr"
- `inputs`: 包含 WAV 和 WAV_LENS 输入
- `outputs`: 请求 TRANSCRIPTS 输出

**响应参数：**
- `outputs`: 包含 TRANSCRIPTS 输出

#### 3.2.2 Python 客户端示例

```python
import numpy as np
import tritonclient.grpc as grpcclient
import librosa

# 加载音频
def load_audio(audio_path, target_sr=16000):
    audio, sr = librosa.load(audio_path, sr=target_sr)
    return audio.astype(np.float32)

# 初始化客户端
client = grpcclient.InferenceServerClient(url="localhost:8001")

# 准备音频数据
audio = load_audio("path/to/audio.wav")

# 准备输入
wav_input = grpcclient.InferInput("WAV", [1, len(audio)], "FP32")
wav_input.set_data_from_numpy(audio.reshape(1, -1))

wav_lens_input = grpcclient.InferInput("WAV_LENS", [1, 1], "INT32")
wav_lens_input.set_data_from_numpy(np.array([[len(audio)]], dtype=np.int32))

# 准备输出
transcript_output = grpcclient.InferRequestedOutput("TRANSCRIPTS")

# 发送请求
response = client.infer(
    model_name="funasr",
    inputs=[wav_input, wav_lens_input],
    outputs=[transcript_output]
)

# 处理响应
transcript = response.as_numpy("TRANSCRIPTS")[0][0]
if isinstance(transcript, bytes):
    transcript = transcript.decode("utf-8")

print("识别结果:", transcript)
```

## 4. 输入输出格式

### 4.1 输入参数

| 参数名 | 数据类型 | 形状 | 描述 |
|--------|----------|------|------|
| WAV | FP32 | [1, N] | 音频数据，N 为音频样本数 |
| WAV_LENS | INT32 | [1, 1] | 音频长度，即 N 值 |

### 4.2 输出参数

| 参数名 | 数据类型 | 形状 | 描述 |
|--------|----------|------|------|
| TRANSCRIPTS | BYTES | [1, 1] | 识别出的文本 |

### 4.3 音频格式要求

- 采样率：16kHz
- 声道：单声道
- 格式：PCM 浮点数据

## 5. 错误处理

### 5.1 常见错误

| 错误码 | 描述 | 可能原因 |
|--------|------|----------|
| 400 | Bad Request | 请求格式错误，如输入参数缺失或格式不正确 |
| 500 | Internal Server Error | 服务器内部错误，如模型加载失败 |
| 503 | Service Unavailable | 服务不可用，如模型未就绪 |

### 5.2 错误响应示例

```json
{
  "error": "inference request failed: input tensor WAV has incorrect shape"
}
```

## 6. 性能指标

### 6.1 基准测试结果

在单 NVIDIA H20 GPU 上的测试结果：

| 并发度 | 字符错误率 (CER) | 处理时间 | P50 延迟 | 平均延迟 | 实时因子 (RTF) |
|--------|-----------------|---------|---------|---------|---------------|
| 8      | 7.04%           | 44.56s  | 450.99ms| 458.17ms| 0.0126        |
| 16     | 7.00%           | 27.96s  | 533.36ms| 571.19ms| 0.0079        |
| 32     | 7.07%           | 24.51s  | 952.93ms| 1001.56ms| 0.0069       |

*注：RTF (实时因子) = 处理时间 / 音频时长，值越小越好。测试使用 SPEECHIO 007 测试集（约 1 小时音频）。*

### 6.2 最佳实践

- 对于实时应用，建议使用较低的并发度（如 8-16）以获得更低的延迟
- 对于批量处理，建议使用较高的并发度（如 32）以获得更高的吞吐量
- 音频长度建议控制在合理范围内，过长的音频可能会导致延迟增加

## 7. 测试工具

### 7.1 内置测试客户端

项目提供了现成的测试客户端脚本：

#### HTTP 客户端测试

```bash
python3 triton_server/http_client.py --audio triton_server/assets/zh.wav --server localhost:8000 --model funasr
```

#### gRPC 客户端测试

```bash
python3 triton_server/grpc_client.py --audio triton_server/assets/zh.wav --server localhost:8001 --model funasr
```

### 7.2 批量测试

```bash
./triton_server/benchmark_client.sh
```

## 8. 多语言支持

### 8.1 支持语言

- **基础模型**：中文、英文、日文
- **多语言模型**：中文、英文、粤语、日文、韩文、越南语、印尼语、泰语、马来语、菲律宾语、阿拉伯语、印地语、保加利亚语、克罗地亚语、捷克语、丹麦语、荷兰语、爱沙尼亚语、芬兰语、希腊语、匈牙利语、爱尔兰语、拉脱维亚语、立陶宛语、马耳他语、波兰语、葡萄牙语、罗马尼亚语、斯洛伐克语、斯洛文尼亚语、瑞典语

### 8.2 语言指定

在使用多语言模型时，可通过请求参数指定语言。

## 9. 部署配置

### 9.1 模型配置

模型配置文件位于 `triton_server/model_repo_funasr/funasr/config.pbtxt`，关键参数：

- `max_batch_size`：最大批处理大小
- `preferred_batch_size`：首选批处理大小
- `max_queue_delay_microseconds`：批处理最大等待时间

### 9.2 服务配置

可通过 `run_server.sh` 脚本的参数调整服务配置：

```bash
./run_server.sh --http-port 8000 --grpc-port 8001 --metrics-port 8002 --gpu 0
```

## 10. 常见问题

### 10.1 服务启动失败

- 检查 GPU 是否可用：`nvidia-smi`
- 检查 Docker 是否正确安装 NVIDIA Container Toolkit
- 检查端口是否被占用

### 10.2 识别结果不准确

- 确保音频质量良好，无过多噪音
- 确保音频采样率为 16kHz
- 对于特定领域的内容，可考虑使用热词列表

### 10.3 服务响应缓慢

- 检查 GPU 使用率，避免过载
- 调整批处理大小和并发度
- 对于长音频，考虑分段处理

## 11. 版本信息

| 版本 | 模型 | 特性 |
|------|------|------|
| 25.06 | Fun-ASR-Nano-2512 | 基础模型，支持中英文 |
| 25.06 | Fun-ASR-MLT-Nano-2512 | 多语言模型，支持 30+ 语言 |

## 12. 联系方式

如有问题或建议，请通过以下方式联系：

- GitHub 仓库：[Fun-ASR-vllm](https://github.com/yuekaizhang/Fun-ASR-vllm)
- 原项目：[Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR)

---

*文档更新时间：2026-02-05*
