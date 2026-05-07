"""
针对吞音问题，对单条样本跑多种降噪强度对比。

策略：
  - DFN3 自带的 atten_lim_db 参数：限制最大衰减量（默认 100dB → 几乎无限）
  - 干湿混合：输出 = α·降噪 + (1-α)·原始

对吞音敏感的弱辅音（汉字塞音如"大"的爆破段、擦音）通常被误判为噪声，
限制衰减量可以保留这些细节。
"""
import argparse
import os
import sys
import types
import warnings

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

# Patch torchaudio.backend
import torchaudio
if not hasattr(torchaudio, "backend"):
    torchaudio.backend = types.ModuleType("torchaudio.backend")
    torchaudio.backend.common = types.ModuleType("torchaudio.backend.common")
    class AudioMetaData: pass
    torchaudio.backend.common.AudioMetaData = AudioMetaData
    sys.modules["torchaudio.backend"] = torchaudio.backend
    sys.modules["torchaudio.backend.common"] = torchaudio.backend.common

import torch
from df.enhance import enhance, init_df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TARGET_SR = 16000


def denoise(model, df_state, x_16k, atten_lim_db=None):
    x_48k = librosa.resample(x_16k, orig_sr=TARGET_SR, target_sr=df_state.sr())
    tensor = torch.from_numpy(x_48k).unsqueeze(0)
    kwargs = {"atten_lim_db": atten_lim_db} if atten_lim_db is not None else {}
    enhanced = enhance(model, df_state, tensor, **kwargs).squeeze(0).numpy()
    return librosa.resample(enhanced, orig_sr=df_state.sr(), target_sr=TARGET_SR).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
        default=os.path.join(BASE_DIR, "recordings_cleaned", "green",
                             "65b5ff9ac12e469ab6754e34db61b1c4.wav"),
        help="测试文件路径")
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "test_denoise_strength"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    name = os.path.basename(args.input).replace(".wav", "")[:12]

    # 加载（兼容 MP4 容器伪 wav）
    try:
        x, sr = sf.read(args.input, dtype="float32")
    except Exception:
        x, sr = librosa.load(args.input, sr=None, mono=True)
        x = x.astype(np.float32)
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    print(f"输入：{name}  时长 {len(x)/TARGET_SR:.2f}s\n")

    print("加载 DFN3...")
    model, df_state, _ = init_df()

    # 配置：(后缀, atten_lim_db, dry_wet 比例, 是否做RMS增益匹配)
    configs = [
        ("00_原始",                       None, 0.0, False),
        ("06_干湿50%(原版)",              None, 0.5, False),
        ("06a_干湿50%+RMS增益匹配",       None, 0.5, True),
        ("06b_干湿70%+RMS增益匹配",       None, 0.7, True),
        ("06c_干湿50%+峰值归一化_-3dBFS", None, 0.5, "peak"),
    ]

    print(f"\n生成 {len(configs)} 个版本...")
    full_dn = denoise(model, df_state, x)
    target_rms = np.sqrt(np.mean(x ** 2))

    for suffix, atten, wet, gain_mode in configs:
        if suffix == "00_原始":
            out = x.copy()
        elif atten is not None:
            out = denoise(model, df_state, x, atten_lim_db=atten)
        else:
            n = min(len(x), len(full_dn))
            out = (wet * full_dn[:n] + (1 - wet) * x[:n]).astype(np.float32)

        # 增益匹配
        if gain_mode is True:
            cur_rms = np.sqrt(np.mean(out ** 2)) + 1e-12
            out = out * (target_rms / cur_rms)
            # 防削波
            peak = np.max(np.abs(out))
            if peak > 0.99:
                out = out * (0.99 / peak)
        elif gain_mode == "peak":
            peak = np.max(np.abs(out)) + 1e-12
            out = out * (10 ** (-3/20) / peak)  # -3 dBFS

        out = out.astype(np.float32)
        path = os.path.join(args.out, f"{suffix}.wav")
        sf.write(path, out, TARGET_SR)

        rms = 20 * np.log10(np.sqrt(np.mean(out ** 2)) + 1e-12)
        peak_db = 20 * np.log10(np.max(np.abs(out)) + 1e-12)
        print(f"  {suffix:<32}  RMS={rms:>5.1f}dB  Peak={peak_db:>5.1f}dB")

    print(f"\n输出目录: {args.out}")
    print(f"\n推荐按顺序对比 02→06，找到「噪声压住但'大'字没吞」的版本")


if __name__ == "__main__":
    main()
