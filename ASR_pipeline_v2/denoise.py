"""
DeepFilterNet3 降噪模块（适配校园嘈杂场景，避免吞音）

经过 A/B 测试确定的最优配置：
  - DFN3 满强度推理
  - 50% 干湿混合（保留弱辅音、爆破段细节）
  - RMS 增益匹配 + 防削波（保持原始响度）

用法：
    from denoise import Denoiser
    dn = Denoiser()                          # 单例加载（首次约 0.1s）
    y = dn(x_16k)                            # x_16k: float32 numpy, mono, 16kHz
"""
import sys
import types
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ===== Patch torchaudio.backend missing in 2.11 =====
import torchaudio
if not hasattr(torchaudio, "backend"):
    torchaudio.backend = types.ModuleType("torchaudio.backend")
    torchaudio.backend.common = types.ModuleType("torchaudio.backend.common")
    class AudioMetaData: pass
    torchaudio.backend.common.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend"] = torchaudio.backend
    sys.modules["torchaudio.backend.common"] = torchaudio.backend.common

import librosa
import torch
from df.enhance import enhance, init_df

TARGET_SR = 16000


class Denoiser:
    """惰性加载 DFN3，第一次调用时初始化模型。"""

    _instance = None  # 单例

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, dry_wet=0.5, peak_limit=0.99):
        if self._initialized:
            return
        self.dry_wet = dry_wet      # 降噪输出占比（0=纯原始, 1=纯降噪）
        self.peak_limit = peak_limit
        self.model = None
        self.df_state = None
        self._initialized = True

    def _ensure_loaded(self):
        if self.model is None:
            self.model, self.df_state, _ = init_df()

    def __call__(self, x_16k: np.ndarray) -> np.ndarray:
        """输入：float32 numpy, mono, 16kHz；输出：同形状降噪音频。"""
        self._ensure_loaded()
        x = x_16k.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)

        # 16k → 48k → DFN3 → 16k
        sr_model = self.df_state.sr()
        x_48k = librosa.resample(x, orig_sr=TARGET_SR, target_sr=sr_model)
        tensor = torch.from_numpy(x_48k).unsqueeze(0)
        denoised_48k = enhance(self.model, self.df_state, tensor).squeeze(0).numpy()
        denoised = librosa.resample(denoised_48k, orig_sr=sr_model, target_sr=TARGET_SR)

        # 长度对齐
        n = min(len(x), len(denoised))
        x, denoised = x[:n], denoised[:n]

        # 干湿混合
        out = self.dry_wet * denoised + (1 - self.dry_wet) * x

        # RMS 增益匹配回原始响度
        target_rms = np.sqrt(np.mean(x ** 2)) + 1e-12
        cur_rms = np.sqrt(np.mean(out ** 2)) + 1e-12
        out = out * (target_rms / cur_rms)

        # 防削波
        peak = np.max(np.abs(out))
        if peak > self.peak_limit:
            out = out * (self.peak_limit / peak)

        return out.astype(np.float32)


def denoise(x_16k: np.ndarray) -> np.ndarray:
    """函数式 API（内部走单例）。"""
    return Denoiser()(x_16k)
