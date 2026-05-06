"""
统一的设备（GPU/CPU）管理工具。

- get_device(): 返回 "cuda" 或 "cpu"，自动检测
- get_voice_encoder(): 返回缓存的 Resemblyzer VoiceEncoder（避免每次加载）

环境变量 / 命令行覆盖:
  ASR_PIPELINE_DEVICE=cpu   # 强制 CPU
  ASR_PIPELINE_DEVICE=cuda  # 强制 GPU
默认: 自动检测，有 CUDA 用 CUDA，否则 CPU
"""
import os

_device = None
_voice_encoder = None


def get_device():
    """返回 'cuda' 或 'cpu'。第一次调用做检测，后续返回缓存。"""
    global _device
    if _device is not None:
        return _device

    forced = os.environ.get("ASR_PIPELINE_DEVICE", "").lower()
    if forced in ("cpu", "cuda"):
        _device = forced
        return _device

    try:
        import torch
        if torch.cuda.is_available():
            _device = "cuda"
        else:
            _device = "cpu"
    except ImportError:
        _device = "cpu"
    return _device


def device_info():
    """打印当前设备信息"""
    d = get_device()
    if d == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            return f"cuda:0 ({name}, sm_{cap[0]}{cap[1]})"
        except Exception:
            return "cuda:0"
    return "cpu"


def get_voice_encoder():
    """缓存 Resemblyzer VoiceEncoder，第一次创建后复用"""
    global _voice_encoder
    if _voice_encoder is None:
        from resemblyzer import VoiceEncoder
        device = get_device()
        try:
            _voice_encoder = VoiceEncoder(device=device, verbose=False)
        except TypeError:
            # 老版本 resemblyzer 可能不接受 device 参数
            _voice_encoder = VoiceEncoder(verbose=False)
    return _voice_encoder


if __name__ == "__main__":
    print(f"Device: {device_info()}")
    print(f"forced (env ASR_PIPELINE_DEVICE): {os.environ.get('ASR_PIPELINE_DEVICE', '<unset>')}")
