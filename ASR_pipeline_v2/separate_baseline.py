"""
对 samples_test/ 的所有音频做：
  1. 加载（PyAV → 16k 单声道）
  2. 当前 ONNX 分离模型 + 增强版后处理（RMS归一/谱减/预加重/Wiener）
  3. 用 FunASR 对 mix / 轨1 / 轨2 各跑一次 ASR
  4. 输出报告

输出: output_separated_enhanced/
"""
import os
import sys
import glob
import time
import numpy as np
import soundfile as sf
import onnxruntime as ort
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio, save_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "samples_test")
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.onnx"))
OUT_DIR = os.path.join(BASE_DIR, "output_separated_enhanced")
SR = 16000

# 后处理开关（与 separate_enhanced.py 保持一致）
USE_DENOISE = True
USE_PREEMPHASIS = True
USE_WIENER = True
WIENER_POWER = 2.0


# ---- 后处理工具 ----
def rms_normalize(x, target_rms=0.1):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-9
    scale = target_rms / rms
    return x * scale, scale


def preemphasis(x, c=0.97):
    return np.append(x[0], x[1:] - c * x[:-1]).astype(np.float32)


def deemphasis(x, c=0.97):
    y = np.zeros_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = x[i] + c * y[i - 1]
    return y.astype(np.float32)


def stft(x, n_fft=512, hop=128):
    win = np.hanning(n_fft).astype(np.float32)
    pad = (n_fft - hop) // 2
    xp = np.pad(x, (pad, pad + n_fft))
    frames = sliding_window_view(xp, n_fft)[::hop] * win
    return np.fft.rfft(frames, axis=1), len(xp), pad, win


def istft(spec, total_len, pad, win, target_len, n_fft=512, hop=128):
    frames = np.fft.irfft(spec, n=n_fft, axis=1) * win
    out = np.zeros(total_len, dtype=np.float32)
    norm = np.zeros(total_len, dtype=np.float32)
    for i, f in enumerate(frames):
        s = i * hop
        out[s:s + n_fft] += f
        norm[s:s + n_fft] += win ** 2
    return (out / np.maximum(norm, 1e-6))[pad:pad + target_len].astype(np.float32)


def spectral_denoise(x, n_fft=512, hop=128, noise_frames=10):
    spec, total, pad, win = stft(x, n_fft, hop)
    mag, phase = np.abs(spec), np.angle(spec)
    noise_mag = np.mean(mag[:noise_frames], axis=0, keepdims=True)
    sub = np.maximum(mag - 1.5 * noise_mag, 0.05 * mag)
    return istft(sub * np.exp(1j * phase), total, pad, win, len(x), n_fft, hop)


def wiener_refine(mix, sources, power=2.0):
    n = min(len(mix), min(len(s) for s in sources))
    mix = mix[:n]
    sources = [s[:n] for s in sources]
    mix_spec, total, pad, win = stft(mix)
    src_specs = [stft(s)[0] for s in sources]
    mags_p = [np.abs(S) ** power for S in src_specs]
    denom = np.sum(mags_p, axis=0) + 1e-8
    masks = [m / denom for m in mags_p]
    return [istft(m * mix_spec, total, pad, win, n) for m in masks]


def db(x):
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


# ---- 单文件分离流程 ----
def separate_one(session, raw):
    x = raw.copy()
    if USE_DENOISE:
        x = spectral_denoise(x)
    if USE_PREEMPHASIS:
        x = preemphasis(x)
    x_norm, scale = rms_normalize(x, 0.1)

    in_name = session.get_inputs()[0].name
    outs = session.run(None, {in_name: np.expand_dims(x_norm, 0)})
    sources = [o.squeeze().astype(np.float32) / scale for o in outs]

    if USE_PREEMPHASIS:
        sources = [deemphasis(s) for s in sources]
    if USE_WIENER:
        sources = wiener_refine(raw, sources, WIENER_POWER)

    # 防削波
    out = []
    for s in sources:
        peak = np.max(np.abs(s))
        if peak > 0.99:
            s = s * (0.99 / peak)
        out.append(s)
    return out


# ---- 评估 ----
def evaluate_separation(mix, sources):
    """无参考三档评估"""
    n = min(len(mix), min(len(s) for s in sources))
    mix, sources = mix[:n], [s[:n] for s in sources]
    recon = np.sum(sources, axis=0)
    alpha = np.dot(mix, recon) / (np.dot(recon, recon) + 1e-12)
    res = mix - alpha * recon
    snr = db(mix) - db(res)

    energies = [np.mean(s ** 2) for s in sources]
    total = sum(energies) + 1e-12
    ratios = [e / total for e in energies]

    return {
        "alpha": float(alpha),
        "重建SNR(dB)": float(snr),
        "能量比": [round(r, 3) for r in ratios],
    }


def speaker_similarity(sources):
    try:
        from resemblyzer import preprocess_wav
        from device_utils import get_voice_encoder
        enc = get_voice_encoder()  # 首次加载后缓存复用
        embs = [enc.embed_utterance(preprocess_wav(s, source_sr=SR)) for s in sources]
        e0, e1 = embs[0], embs[1]
        cos = float(np.dot(e0, e1) / (np.linalg.norm(e0) * np.linalg.norm(e1) + 1e-12))
        return cos
    except Exception as e:
        return f"err: {e}"


# ---- 主流程 ----
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.m4a")) +
                   glob.glob(os.path.join(DATASET_DIR, "*.wav")))
    if not files:
        print(f"未找到音频文件: {DATASET_DIR}")
        return

    print(f"找到 {len(files)} 个音频文件\n")

    print("加载分离模型...")
    session = ort.InferenceSession(MODEL_PATH)

    print("加载 FunASR (Paraformer-zh + cam++ + ct-punc)...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", spk_model="cam++",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    def transcribe(samples):
        result = asr.generate(input=samples, fs=SR, batch_size_s=300, return_spk_res=True)
        parts = []
        for item in result:
            if item.get("sentence_info"):
                for sent in item["sentence_info"]:
                    parts.append(sent.get("text", "").strip())
            else:
                parts.append(item.get("text", "").strip())
        return "".join(parts).strip()

    print()

    # ---- 处理每个文件 ----
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        print("=" * 78)
        print(f"📁 {os.path.basename(path)}")

        t0 = time.time()
        mix, sr = load_audio(path, verbose=True)
        dur = len(mix) / sr
        print(f"   时长 {dur:.2f}s, RMS {np.sqrt(np.mean(mix**2)):.4f}")

        # 分离
        sources = separate_one(session, mix)
        print(f"   分离耗时 {time.time()-t0:.1f}s")

        # 保存
        save_audio(os.path.join(OUT_DIR, f"{name}_mix.wav"), mix, SR)
        for i, s in enumerate(sources):
            save_audio(os.path.join(OUT_DIR, f"{name}_轨{i+1}.wav"), s, SR)

        # 评估
        metrics = evaluate_separation(mix, sources)
        sim = speaker_similarity(sources)
        print(f"   重建 SNR: {metrics['重建SNR(dB)']:.1f} dB | "
              f"能量比 {metrics['能量比']} | 说话人相似度 {sim if isinstance(sim, str) else f'{sim:.3f}'}")

        # ASR
        print(f"\n   --- ASR (FunASR) ---")
        try:
            t0 = time.time()
            mix_text = transcribe(mix)
            print(f"   [混合]   ({time.time()-t0:.1f}s) {mix_text}")
        except Exception as e:
            print(f"   [混合]   err: {e}")

        for i, s in enumerate(sources):
            try:
                t0 = time.time()
                txt = transcribe(s)
                print(f"   [轨{i+1}]   ({time.time()-t0:.1f}s) {txt}")
            except Exception as e:
                print(f"   [轨{i+1}]   err: {e}")
        print()

    print("=" * 78)
    print(f"分离结果已保存到: {OUT_DIR}/")


if __name__ == "__main__":
    main()
