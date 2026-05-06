#!/usr/bin/env python3
"""
Fun-ASR vLLM API 综合测试脚本
基于 API_DOCUMENTATION.md 中的 API 文档
"""

import argparse
import numpy as np
import time
import json
from typing import Optional, Dict, Any

# 尝试导入所需库
try:
    import librosa
except ImportError:
    print("错误: 未找到 librosa 库。请使用 'pip install librosa' 安装")
    exit(1)

try:
    import tritonclient.http as httpclient
    import tritonclient.grpc as grpcclient
except ImportError:
    print("错误: 未找到 tritonclient 库。请使用 'pip install tritonclient[all]' 安装")
    exit(1)


def load_audio(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """
    加载音频文件并在必要时重采样。
    
    参数:
        audio_path: 音频文件路径
        target_sr: 目标采样率
        
    返回:
        音频样本，类型为 float32 numpy 数组
    """
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr)
        return audio.astype(np.float32)
    except Exception as e:
        print(f"加载音频错误: {e}")
        raise


def check_server_health(http_url: str) -> bool:
    """
    检查 Triton 服务器是否健康。
    
    参数:
        http_url: 服务器的 HTTP URL
        
    返回:
        如果服务器健康返回 True，否则返回 False
    """
    try:
        client = httpclient.InferenceServerClient(url=http_url)
        health = client.is_server_ready()
        print(f"服务器健康检查: {'正常' if health else '失败'}")
        return health
    except Exception as e:
        print(f"检查服务器健康状态错误: {e}")
        return False


def check_model_status(http_url: str, model_name: str) -> Dict[str, Any]:
    """
    检查模型状态。
    
    参数:
        http_url: 服务器的 HTTP URL
        model_name: 模型名称
        
    返回:
        模型状态信息
    """
    try:
        import requests
        url = f"http://{http_url}/v2/models/{model_name}"
        response = requests.get(url)
        if response.status_code == 200:
            status = response.json()
            print(f"模型状态: {status.get('state', '未知')}")
            return status
        else:
            print(f"检查模型状态错误: {response.status_code}")
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        print(f"检查模型状态错误: {e}")
        return {"error": str(e)}


def test_http_api(audio_path: str, server_url: str, model_name: str) -> Optional[str]:
    """
    测试 HTTP API。
    
    参数:
        audio_path: 音频文件路径
        server_url: 服务器的 HTTP URL
        model_name: 模型名称
        
    返回:
        转录文本，如果失败则返回 None
    """
    print("\n=== 测试 HTTP API ===")
    
    try:
        # 加载音频
        start_time = time.time()
        audio = load_audio(audio_path)
        load_time = time.time() - start_time
        print(f"音频加载时间: {load_time:.3f} 秒")
        print(f"音频长度: {len(audio)} 样本 ({len(audio)/16000:.2f} 秒)")
        
        # 创建客户端
        client = httpclient.InferenceServerClient(url=server_url)
        
        # 准备输入
        wav_input = httpclient.InferInput("WAV", [1, len(audio)], "FP32")
        wav_input.set_data_from_numpy(audio.reshape(1, -1))
        
        wav_lens_input = httpclient.InferInput("WAV_LENS", [1, 1], "INT32")
        wav_lens_input.set_data_from_numpy(np.array([[len(audio)]], dtype=np.int32))
        
        # 准备输出
        transcript_output = httpclient.InferRequestedOutput("TRANSCRIPTS")
        
        # 运行推理
        start_time = time.time()
        response = client.infer(
            model_name=model_name,
            inputs=[wav_input, wav_lens_input],
            outputs=[transcript_output]
        )
        infer_time = time.time() - start_time
        print(f"推理完成时间: {infer_time:.3f} 秒")
        print(f"实时因子 (RTF): {infer_time/(len(audio)/16000):.4f}")
        
        # 获取结果
        transcript = response.as_numpy("TRANSCRIPTS")[0][0]
        if isinstance(transcript, bytes):
            transcript = transcript.decode("utf-8")
        
        print(f"转录结果: {transcript}")
        return transcript
        
    except Exception as e:
        print(f"测试 HTTP API 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_error_handling(http_url: str) -> None:
    """
    测试错误处理能力。
    
    参数:
        http_url: 服务器的 HTTP URL
    """
    print("\n=== 测试错误处理 ===")
    
    try:
        # 创建客户端
        client = httpclient.InferenceServerClient(url=http_url)
        
        # 测试无效音频长度
        print("测试无效音频长度...")
        
        # 创建虚拟音频
        audio = np.zeros(1000, dtype=np.float32)
        
        # 准备错误长度的输入
        wav_input = httpclient.InferInput("WAV", [1, len(audio)], "FP32")
        wav_input.set_data_from_numpy(audio.reshape(1, -1))
        
        # 设置错误长度
        wav_lens_input = httpclient.InferInput("WAV_LENS", [1, 1], "INT32")
        wav_lens_input.set_data_from_numpy(np.array([[999999]], dtype=np.int32))  # 错误长度
        
        # 准备输出
        transcript_output = httpclient.InferRequestedOutput("TRANSCRIPTS")
        
        # 运行推理
        response = client.infer(
            model_name="funasr",
            inputs=[wav_input, wav_lens_input],
            outputs=[transcript_output]
        )
        
        print("错误: 预期会出错但获得了响应")
        
    except Exception as e:
        print(f"收到预期的错误: {type(e).__name__}")
        print(f"错误信息: {e}")


def main():
    """
    Fun-ASR vLLM API 测试主函数
    """
    parser = argparse.ArgumentParser(description="Fun-ASR vLLM API 综合测试脚本")
    
    # 必填参数
    parser.add_argument(
        "--audio",
        type=str,
        required=True,
        help="测试音频文件路径"
    )
    
    # 可选参数
    parser.add_argument(
        "--http-server",
        type=str,
        default="172.16.254.126:8000",
        help="HTTP 服务器 URL (默认: localhost:8000)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="funasr",
        help="模型名称 (默认: funasr)"
    )
    
    parser.add_argument(
        "--api",
        type=str,
        choices=["http"],
        default="http",
        help="要测试的 API (默认: http)"
    )
    
    parser.add_argument(
        "--test-error",
        action="store_true",
        help="测试错误处理"
    )
    
    args = parser.parse_args()
    
    print("====================================")
    print("Fun-ASR vLLM API 测试脚本")
    print("====================================")
    print(f"音频文件: {args.audio}")
    print(f"HTTP 服务器: {args.http_server}")
    print(f"模型: {args.model}")
    print(f"测试 API: {args.api}")
    print(f"测试错误处理: {args.test_error}")
    print("====================================")
    
    # 检查服务器健康状态
    print("\n=== 检查服务器健康状态 ===")
    health = check_server_health(args.http_server)
    
    if not health:
        print("服务器状态不健康。请检查服务器是否正在运行。")
        return
    
    # 检查模型状态
    print("\n=== 检查模型状态 ===")
    model_status = check_model_status(args.http_server, args.model)
    
    if "error" in model_status:
        print(f"模型状态检查失败: {model_status['error']}")
        return
    
    # 测试 API
    results = {}
    
    if args.api in ["http", "both"]:
        results["http"] = test_http_api(args.audio, args.http_server, args.model)

    # 测试错误处理
    if args.test_error:
        test_error_handling(args.http_server)
    
    # 总结
    print("\n=== 测试总结 ===")
    
    if "http" in results:
        status = "通过" if results["http"] else "失败"
        print(f"HTTP API 测试: {status}")
        if results["http"]:
            print(f"HTTP 转录结果: {results['http']}")
    
    print("\n=== 测试完成 ===")


if __name__ == "__main__":
    main()
