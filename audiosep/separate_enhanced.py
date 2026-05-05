"""
增强版分离脚本 —— 不改模型，仅在输入/输出两端做处理来提升效果。

主要改进：
  1. 输入 RMS 归一化       —— 让模型工作在它训练时熟悉的电平
  2. 输入轻量预加重        —— 提高高频清晰度（可关）
  3. 能量回贴 (α-projection) —— 把每路输出缩放到与 mix 同尺度
  4. Wiener 软掩码精修     —— 用模型输出做幅度比，再乘回 mix STFT
                             保证 s1+s2≈mix，抑制模型伪影
  5. 可选输入降噪          —— 简单谱减法（无外部依赖）
"""
import os
import numpy as np
import soundfile as sf
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "model.onnx")
INPUT_WAV = os.path.join(BASE_DIR, "2in1.wav")
SAMPLE_RATE = 16000

# 后处理开关
USE_PREEMPHASIS = True       # 输入预加重
USE_SPEC_DENOISE = True      # 输入端谱减降噪
USE_ENERGY_PROJECT = True    # 输出能量回贴
USE_WIENER_MASK = True       # Wiener 软掩码精修
WIENER_POWER = 2.0           # 掩码指数（1=幅度，2=功率，2 一般更干净）


# -------- 工具 --------
def load_audio(path):
    data, sr = sf.read(path)
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"采样率 {sr} != {SAMPLE_RATE}")
    return data.astype(np.float32)


def rms_normalize(x, target_rms=0.1):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    return x * (target_rms / rms), target_rms / rms


def preemphasis(x, coef=0.97):
    return np.append(x[0], x[1:] - coef * x[:-1]).astype(np.float32)


def deemphasis(x, coef=0.97):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + coef * y[i - 1]
    return y.astype(np.float32)


# -------- 1. 输入端谱减降噪 --------
def spectral_denoise(x, n_fft=512, hop=128, noise_frames=10):
    """
    用前 noise_frames 帧估计噪声谱，做经典谱减。
    对持续背景噪声有效，对突发噪声无效。
    """
    from numpy.lib.stride_tricks import sliding_window_view

    # STFT
    win = np.hanning(n_fft).astype(np.float32)
    pad = (n_fft - hop) // 2
    xp = np.pad(x, (pad, pad + n_fft))
    frames = sliding_window_view(xp, n_fft)[::hop] * win
    spec = np.fft.rfft(frames, axis=1)
    mag = np.abs(spec)
    phase = np.angle(spec)

    noise_mag = np.mean(mag[:noise_frames], axis=0, keepdims=True)
    sub = np.maximum(mag - 1.5 * noise_mag, 0.05 * mag)

    # iSTFT (overlap-add)
    new_spec = sub * np.exp(1j * phase)
    new_frames = np.fft.irfft(new_spec, n=n_fft, axis=1) * win
    out = np.zeros(len(xp), dtype=np.float32)
    norm = np.zeros(len(xp), dtype=np.float32)
    for i, f in enumerate(new_frames):
        s = i * hop
        out[s:s + n_fft] += f
        norm[s:s + n_fft] += win ** 2
    out = out / np.maximum(norm, 1e-6)
    return out[pad:pad + len(x)].astype(np.float32)


# -------- 4. Wiener 软掩码精修 --------
def wiener_refine(mix, sources, n_fft=512, hop=128, power=2.0):
    """
    把模型输出当作各源能量分布估计，构造软掩码 m_i = |S_i|^p / Σ|S_j|^p
    再用 m_i * Mix 复频谱重建，输出严格满足 Σ s_i = mix。
    """
    n = min(len(mix), min(len(s) for s in sources))
    mix = mix[:n]
    sources = [s[:n] for s in sources]

    win = np.hanning(n_fft).astype(np.float32)
    pad = (n_fft - hop) // 2

    def stft(x):
        from numpy.lib.stride_tricks import sliding_window_view
        xp = np.pad(x, (pad, pad + n_fft))
        frames = sliding_window_view(xp, n_fft)[::hop] * win
        return np.fft.rfft(frames, axis=1), len(xp)

    def istft(spec, total_len):
        frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
        out = np.zeros(total_len, dtype=np.float32)
        norm = np.zeros(total_len, dtype=np.float32)
        for i, f in enumerate(frames):
            s = i * hop
            out[s:s + n_fft] += f
            norm[s:s + n_fft] += win ** 2
        return (out / np.maximum(norm, 1e-6))[pad:pad + n].astype(np.float32)

    mix_spec, total = stft(mix)
    src_specs = [stft(s)[0] for s in sources]
    mags_p = [np.abs(S) ** power for S in src_specs]
    denom = np.sum(mags_p, axis=0) + 1e-8
    masks = [m / denom for m in mags_p]

    refined = [istft(mask * mix_spec, total) for mask in masks]
    return refined


# -------- 3. 能量回贴 --------
def energy_project(mix, sources):
    """每路按 <s, mix>/<s, s> 缩放到与 mix 同尺度的最优投影"""
    out = []
    n = min(len(mix), min(len(s) for s in sources))
    mix = mix[:n]
    for s in sources:
        s = s[:n]
        alpha = np.dot(s, mix) / (np.dot(s, s) + 1e-12)
        out.append(s * alpha)
    return out


# -------- 主流程 --------
def main():
    print("加载音频...")
    raw = load_audio(INPUT_WAV)
    print(f"  原始长度 {len(raw)} samples，RMS {np.sqrt(np.mean(raw ** 2)):.4f}")

    x = raw.copy()

    if USE_SPEC_DENOISE:
        print("[1/4] 输入谱减降噪...")
        x = spectral_denoise(x)

    if USE_PREEMPHASIS:
        print("[2/4] 输入预加重...")
        x = preemphasis(x)

    print("[*]   RMS 归一化...")
    x_norm, scale = rms_normalize(x, target_rms=0.1)

    print("[3/4] 模型推理...")
    session = ort.InferenceSession(MODEL_PATH)
    in_name = session.get_inputs()[0].name
    outputs = session.run(None, {in_name: np.expand_dims(x_norm, 0)})
    sources = [o.squeeze().astype(np.float32) / scale for o in outputs]
    print(f"      分离出 {len(sources)} 路")

    if USE_PREEMPHASIS:
        sources = [deemphasis(s) for s in sources]

    if USE_WIENER_MASK:
        print("[4/4] Wiener 软掩码精修（基于原始 mix）...")
        # 注意用 raw（未降噪未预加重的原始音频）做 mask 应用
        sources = wiener_refine(raw, sources, power=WIENER_POWER)
    elif USE_ENERGY_PROJECT:
        print("[4/4] 能量回贴...")
        sources = energy_project(raw, sources)

    # 防削波
    for i, s in enumerate(sources):
        peak = np.max(np.abs(s))
        if peak > 0.99:
            sources[i] = s * (0.99 / peak)
        out_path = os.path.join(BASE_DIR, f"分离_增强_说话人_{i+1}.wav")
        sf.write(out_path, sources[i], SAMPLE_RATE)
        print(f"  已保存: {out_path}")

    print("\n完成。可用 evaluate.py 对比新输出。")


if __name__ == "__main__":
    main()
