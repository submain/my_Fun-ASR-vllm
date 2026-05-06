from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_download
import os
import ssl
# 设置环境变量使用国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 创建模型目录
os.makedirs("models", exist_ok=True)

print("开始下载基础模型...")
base_model_path = snapshot_download(
    'FunAudioLLM/Fun-ASR-Nano-2512',
    cache_dir='models/funasr-base'
)
print(f"基础模型下载完成: {base_model_path}")

print("\n开始下载 vLLM 模型...")
vllm_model_path = hf_download(
    repo_id="yuekai/Fun-ASR-Nano-2512-vllm",
    local_dir="models/funasr-vllm",
    local_dir_use_symlinks=False
)
print(f"vLLM 模型下载完成: {vllm_model_path}")

print("\n所有模型下载完成！")