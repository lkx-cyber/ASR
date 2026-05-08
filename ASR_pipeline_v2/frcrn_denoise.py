"""
FRCRN_SE_16K 降噪模块（ClearerVoice-Studio）

阿里达摩院出品，中文场景训练，理论上对儿童语音保留更好。
跟 DFN3 的差别：FRCRN 训练数据偏中文，DFN3 偏英文。

注意：
  - 模型只接受文件路径输入，本模块用 tempfile 包了 numpy 接口
  - 速度比 DFN3 慢很多（CPU RTF ≈ 1.8 vs DFN3 0.02）
  - 是「频谱掩码型」模型，对 8k 上采到 16k 的空高频段是安全的（不会乱画）

用法：
    from frcrn_denoise import FRCRNDenoiser
    dn = FRCRNDenoiser()
    y = dn(x_16k)              # x_16k: float32 numpy, mono, 16kHz
"""
import os
import tempfile
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
TARGET_SR = 16000


class FRCRNDenoiser:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.cv = None
        self._initialized = True

    def _ensure_loaded(self):
        if self.cv is None:
            from clearvoice import ClearVoice
            self.cv = ClearVoice(
                task="speech_enhancement",
                model_names=["FRCRN_SE_16K"],
            )

    def __call__(self, x_16k: np.ndarray) -> np.ndarray:
        """16k float32 mono numpy → 16k float32 numpy"""
        self._ensure_loaded()
        x = x_16k.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)

        # 写到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            tmp_path = tf.name
        try:
            sf.write(tmp_path, x, TARGET_SR, subtype="PCM_16")
            out = self.cv(input_path=tmp_path, online_write=False)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        # ClearVoice 返回 (1, N) 或 (N,) 形状
        if isinstance(out, np.ndarray):
            if out.ndim > 1:
                out = out.squeeze()
            return out.astype(np.float32)
        return np.asarray(out, dtype=np.float32).reshape(-1)


def denoise_frcrn(x_16k: np.ndarray) -> np.ndarray:
    return FRCRNDenoiser()(x_16k)
