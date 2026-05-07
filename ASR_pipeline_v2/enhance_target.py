"""
目标说话人增强（Target Speaker Enhancement）

适用场景：PTT 录音里有 1 个主说话人 + 若干弱/强背景人声干扰。
目标：输出"只保留主说话人"的单路音频。

核心思路：
  1. 取前 0.7s 当"声纹锚点"（按按钮的人在录音开始时主导）
  2. 滑窗 (0.75s, 0.25s 步长) 算每个窗与锚点的余弦相似度
  3. 高相似度（同一人）→ mask = 1.0 保留
     低相似度（其他人/喊叫）→ mask 衰减最多 -12dB
  4. 时域平滑后应用，避免 STFT 软掩码的音乐噪声
  5. 增强后 RMS 还原到原始电平，防削波

容错:
  - 锚点静音 → 直接返回原音频（passthrough）
  - 全程相似度都很高 → 单说话人，无需处理
  - 失败时返回原音频 + 错误信息

用法:
    from enhance_target import enhance_target_speaker
    enhanced, info = enhance_target_speaker(raw_audio, sr=16000)
    print(info)  # {"action": "enhance", "min_sim": 0.45, "n_windows": 8}
"""
import os
import sys
import numpy as np
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from device_utils import get_voice_encoder

# 默认参数
ANCHOR_SEC = 0.7        # 锚点时长
WIN_SEC = 0.75          # 滑窗大小
HOP_SEC = 0.25          # 滑窗步长
SIM_KEEP = 0.70         # 相似度 > 此值 → 完全保留
SIM_KILL = 0.40         # 相似度 < 此值 → 最大压制
MAX_SUPPRESS_DB = -12.0 # 最大衰减
SMOOTHING_SEC = 0.1     # 时域平滑窗
MIN_DURATION = 1.5      # 短于此秒数不处理
MIN_RMS = 0.005         # 静音阈值


def _rms(x):
    return float(np.sqrt(np.mean(x ** 2)))


def enhance_target_speaker(
    audio, sr=16000,
    anchor_sec=ANCHOR_SEC, win_sec=WIN_SEC, hop_sec=HOP_SEC,
    sim_keep=SIM_KEEP, sim_kill=SIM_KILL,
    max_suppress_db=MAX_SUPPRESS_DB,
    smoothing_sec=SMOOTHING_SEC,
):
    """
    Args:
        audio: float32 numpy, mono, 任意采样率（建议 16kHz）
        sr: 采样率
    Returns:
        (enhanced_audio, info_dict)
        info_dict = {
            "action": "enhance" | "passthrough" | "skip",
            "reason": str,
            "min_sim": float (可选),
            "n_windows": int (可选),
        }
    """
    audio = np.asarray(audio, dtype=np.float32)
    n = len(audio)
    duration = n / sr

    # 太短 → 不处理
    if duration < MIN_DURATION:
        return audio, {"action": "skip", "reason": f"too short ({duration:.2f}s)"}

    # 锚点
    anchor_n = int(anchor_sec * sr)
    anchor = audio[:anchor_n]
    if _rms(anchor) < MIN_RMS:
        return audio, {"action": "skip", "reason": "anchor silent"}

    from resemblyzer import preprocess_wav
    encoder = get_voice_encoder()

    try:
        anchor_wav = preprocess_wav(anchor, source_sr=sr)
        if len(anchor_wav) < int(0.4 * sr):
            return audio, {"action": "skip", "reason": "anchor too short after VAD"}
        anchor_emb = encoder.embed_utterance(anchor_wav)
        anchor_emb = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-9)
    except Exception as e:
        return audio, {"action": "skip", "reason": f"anchor embed err: {e}"}

    # 滑窗
    win_n = int(win_sec * sr)
    hop_n = int(hop_sec * sr)

    centers = []
    sims = []  # 用 NaN 占位静音段
    for start in range(0, n - win_n + 1, hop_n):
        chunk = audio[start:start + win_n]
        center = start + win_n // 2
        centers.append(center)

        if _rms(chunk) < MIN_RMS:
            sims.append(np.nan)
            continue

        try:
            wav = preprocess_wav(chunk, source_sr=sr)
            if len(wav) < int(0.4 * sr):
                sims.append(np.nan)
                continue
            emb = encoder.embed_utterance(wav)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
            sim = float(np.dot(anchor_emb, emb))
            sims.append(sim)
        except Exception:
            sims.append(np.nan)

    if not centers:
        return audio, {"action": "skip", "reason": "no windows"}

    # 全程都和锚点相似 → 单人，不处理
    valid = [s for s in sims if not np.isnan(s)]
    if len(valid) > 0 and min(valid) > sim_keep + 0.05:
        return audio, {
            "action": "passthrough",
            "reason": "single speaker",
            "min_sim": float(min(valid)),
            "n_windows": len(centers),
        }

    # sim → mask
    min_factor = 10 ** (max_suppress_db / 20.0)  # -12dB ≈ 0.25
    mask_at_centers = np.zeros(len(centers), dtype=np.float32)
    for i, s in enumerate(sims):
        if np.isnan(s):
            # 静音段：保持 1.0（不主动压静音）
            mask_at_centers[i] = 1.0
        else:
            # 线性映射: sim=KEEP → 1.0, sim=KILL → min_factor
            t = (s - sim_kill) / max(sim_keep - sim_kill, 1e-6)
            t = max(0.0, min(1.0, t))
            mask_at_centers[i] = min_factor + (1.0 - min_factor) * t

    # 插值到 sample 级
    centers_arr = np.array(centers, dtype=np.float32)
    sample_idx = np.arange(n, dtype=np.float32)
    sample_mask = np.interp(sample_idx, centers_arr, mask_at_centers).astype(np.float32)

    # 时域平滑（避免突变）
    smooth_n = int(smoothing_sec * sr)
    if smooth_n > 1:
        sample_mask = uniform_filter1d(sample_mask, size=smooth_n, mode="nearest")

    # 应用
    enhanced = audio * sample_mask

    # 还原 RMS（避免听感整体变小）
    raw_rms = _rms(audio)
    enh_rms = _rms(enhanced)
    if enh_rms > 1e-6 and raw_rms > 1e-6:
        enhanced = enhanced * (raw_rms / enh_rms)

    # 防削波
    peak = float(np.max(np.abs(enhanced)))
    if peak > 0.99:
        enhanced = enhanced * (0.99 / peak)

    return enhanced.astype(np.float32), {
        "action": "enhance",
        "min_sim": float(min(valid)) if valid else float("nan"),
        "mean_sim": float(np.nanmean(sims)) if any(not np.isnan(s) for s in sims) else float("nan"),
        "n_windows": len(centers),
        "n_silent": int(sum(1 for s in sims if np.isnan(s))),
    }


if __name__ == "__main__":
    # 单文件 demo
    import sys, json
    if len(sys.argv) < 2:
        print("用法: python enhance_target.py <audio.wav>")
        sys.exit(0)
    from audio_io import load_audio, save_audio
    raw, sr = load_audio(sys.argv[1])
    out, info = enhance_target_speaker(raw, sr)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    save_audio("enhance_target_out.wav", out, sr)
    print("输出: enhance_target_out.wav")
