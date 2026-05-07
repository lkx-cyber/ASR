"""
DeepFilterNet 降噪效果测试

抽 N 条录音（默认 green 和 yellow 各 5 条），跑 DeepFilterNet 降噪。
输出原始/降噪两份 WAV 到 test_denoise/<filename>/ 供人耳对比。

同时打印简单指标：
  - 降噪前后 RMS / 能量变化（dB）
  - 估计噪声底（最低 10% 段能量）变化
  - 处理耗时
"""
import argparse
import os
import random
import sys
import time
import types
import warnings

import numpy as np
import soundfile as sf

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

import torch
from df.enhance import enhance, init_df

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
OUT_DIR = os.path.join(BASE_DIR, "test_denoise")
TARGET_SR = 16000  # 业务采样率


# ---------- 工具 ----------
def load_audio(path, target_sr=TARGET_SR):
    """优先 soundfile，失败则 PyAV，输出单声道 float32。"""
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        import av
        container = av.open(path)
        stream = container.streams.audio[0]
        sr = stream.rate
        chunks = []
        for frame in container.decode(stream):
            chunks.append(frame.to_ndarray())
        container.close()
        data = np.concatenate(chunks, axis=-1).astype(np.float32)
        if data.ndim > 1:
            data = data.mean(axis=0)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return data


def rms_db(x, eps=1e-12):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + eps)


def noise_floor_db(x, frame=int(0.025 * TARGET_SR), eps=1e-12):
    """最低 10% 帧的 RMS 视为噪声底估计。"""
    if len(x) < frame:
        return rms_db(x)
    n = len(x) // frame
    energies = np.array([np.mean(x[i*frame:(i+1)*frame] ** 2) for i in range(n)])
    low10 = np.sort(energies)[: max(1, n // 10)]
    return 10 * np.log10(low10.mean() + eps)


def denoise_one(model, df_state, x_16k):
    """16k 输入 → 重采样到 48k → DFN3 → 重采样回 16k"""
    import librosa
    x_48k = librosa.resample(x_16k, orig_sr=TARGET_SR, target_sr=df_state.sr())
    tensor = torch.from_numpy(x_48k).unsqueeze(0)
    enhanced_48k = enhance(model, df_state, tensor).squeeze(0).numpy()
    enhanced_16k = librosa.resample(enhanced_48k, orig_sr=df_state.sr(), target_sr=TARGET_SR)
    return enhanced_16k.astype(np.float32)


# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-green", type=int, default=5)
    parser.add_argument("--n-yellow", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 抽样
    samples = []
    for tier, n in [("green", args.n_green), ("yellow", args.n_yellow)]:
        tier_dir = os.path.join(CLEANED_DIR, tier)
        if not os.path.isdir(tier_dir):
            print(f"⚠️ 目录不存在: {tier_dir}")
            continue
        files = [f for f in os.listdir(tier_dir) if f.endswith(".wav")]
        picked = random.sample(files, min(n, len(files)))
        samples.extend([(tier, os.path.join(tier_dir, f)) for f in picked])
    print(f"抽样 {len(samples)} 条（green={args.n_green}, yellow={args.n_yellow}）\n")

    # 加载模型
    print("加载 DeepFilterNet3...")
    t0 = time.time()
    model, df_state, _ = init_df()
    print(f"  模型加载耗时: {time.time()-t0:.1f}s\n")

    # 处理每条
    print(f"{'tier':<8} {'file':<22} {'dur':>5}  {'RMS:原→增':>12}  {'噪底:原→增':>12}  {'耗时':>6}")
    print("-" * 80)
    total_proc = 0
    total_dur = 0
    for tier, path in samples:
        fn = os.path.basename(path).replace(".wav", "")
        sub = os.path.join(OUT_DIR, f"{tier}_{fn[:12]}")
        os.makedirs(sub, exist_ok=True)

        x = load_audio(path)
        dur = len(x) / TARGET_SR
        total_dur += dur

        t1 = time.time()
        y = denoise_one(model, df_state, x)
        proc = time.time() - t1
        total_proc += proc

        # 长度对齐
        n = min(len(x), len(y))
        x, y = x[:n], y[:n]

        sf.write(os.path.join(sub, "1_原始.wav"), x, TARGET_SR)
        sf.write(os.path.join(sub, "2_降噪.wav"), y, TARGET_SR)

        rx, ry = rms_db(x), rms_db(y)
        nx, ny = noise_floor_db(x), noise_floor_db(y)
        print(f"{tier:<8} {fn[:20]:<22} {dur:>4.1f}s  "
              f"{rx:>5.1f}→{ry:>5.1f}dB  {nx:>5.1f}→{ny:>5.1f}dB  "
              f"{proc:>5.2f}s")

    rtf = total_proc / total_dur if total_dur else 0
    print("-" * 80)
    print(f"\n汇总：")
    print(f"  总音频时长 {total_dur:.1f}s  总处理耗时 {total_proc:.1f}s  "
          f"实时倍率 RTF={rtf:.2f}（<1 即比实时快）")
    print(f"\n输出目录: {OUT_DIR}")
    print(f"用 Audacity 或 VSCode 听 1_原始.wav 和 2_降噪.wav 对比")


if __name__ == "__main__":
    main()
