"""
量化降噪效果：在原始/降噪样本对上计算多组噪声/语音指标。

指标说明：
  噪声相关（越低越好 / 下降越多代表降噪越强）：
    - noise_floor_db    最低 10% 帧 RMS（背景噪声底估计）
    - hf_noise_db       高频段（4-8kHz）能量，多为宽带噪声
    - lf_rumble_db      低频段（< 150Hz）能量，多为低频隆隆声/HVAC
    - spectral_flatness 非语音段谱平坦度（越低越像「有结构的语音」，越高越像「白噪」）

  信号相关（越高越好 / 提升越多代表语音越突出）：
    - speech_rms_db     语音段（响度高的 50% 帧）RMS
    - snr_db            语音段 RMS - 噪声底（粗略 SNR）
    - hnr_db            谐波-噪声比（仅清浊音段，越高越像清晰人声）
    - crest_db          峰值/RMS（波形动态范围）

输出：
    - 每条样本的指标对比
    - 总体平均/中位数改善量
"""
import os
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
PAIR_DIR = os.path.join(BASE, "test_denoise_asr")
SR = 16000
FRAME = int(0.025 * SR)   # 25ms 帧
HOP = int(0.010 * SR)     # 10ms 跳


def frame_signal(x, frame=FRAME, hop=HOP):
    n = 1 + max(0, (len(x) - frame) // hop)
    out = np.lib.stride_tricks.as_strided(
        x, shape=(n, frame),
        strides=(x.strides[0] * hop, x.strides[0]),
        writeable=False,
    )
    return out.copy()


def db(p, eps=1e-12):
    return 10 * np.log10(np.maximum(p, eps))


# ---------- 各指标 ----------
def noise_floor_db(x):
    """最低 10% 帧的功率"""
    f = frame_signal(x)
    p = (f ** 2).mean(axis=1)
    low = np.sort(p)[: max(1, len(p) // 10)]
    return float(db(low.mean()))


def speech_rms_db(x):
    """最高 50% 帧的 RMS（粗略代表语音段能量）"""
    f = frame_signal(x)
    p = (f ** 2).mean(axis=1)
    hi = np.sort(p)[-len(p) // 2:]
    return float(db(hi.mean()))


def snr_db(x):
    return speech_rms_db(x) - noise_floor_db(x)


def crest_db(x):
    rms = np.sqrt(np.mean(x ** 2)) + 1e-12
    peak = np.max(np.abs(x)) + 1e-12
    return float(20 * np.log10(peak / rms))


def hf_noise_db(x):
    """4-8kHz 带通能量（人声主要在 < 4kHz，此带能量多为噪声）"""
    n = len(x)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1/SR)
    mask = (freqs >= 4000) & (freqs <= 8000)
    p = np.mean(np.abs(fft[mask]) ** 2) / max(1, n)
    return float(db(p))


def lf_rumble_db(x):
    """< 150Hz 低频能量（多为 HVAC、风扇）"""
    n = len(x)
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1/SR)
    mask = freqs <= 150
    p = np.mean(np.abs(fft[mask]) ** 2) / max(1, n)
    return float(db(p))


def spectral_flatness_nonspeech(x):
    """非语音段（最低 30% 帧）的谱平坦度，越低越「无结构」越像有信号残留"""
    f = frame_signal(x)
    p = (f ** 2).mean(axis=1)
    idx = np.argsort(p)[: max(1, len(p) * 3 // 10)]
    quiet = f[idx]
    if len(quiet) == 0:
        return float("nan")
    spec = np.abs(np.fft.rfft(quiet, axis=1)) + 1e-12
    geo = np.exp(np.log(spec).mean(axis=1))
    arith = spec.mean(axis=1)
    sf_vals = geo / arith
    return float(sf_vals.mean())


def hnr_db(x):
    """简化 HNR：在每帧上用自相关找基频峰，估计 H/N。"""
    f = frame_signal(x)
    p = (f ** 2).mean(axis=1)
    voiced_idx = np.where(p > np.median(p))[0]  # 取高于中位数的帧（粗略「有声」）
    if len(voiced_idx) == 0:
        return float("nan")
    hnr_list = []
    for i in voiced_idx:
        frame = f[i] - f[i].mean()
        ac = np.correlate(frame, frame, mode="full")[len(frame)-1:]
        ac = ac / (ac[0] + 1e-12)
        # 在 50-500Hz（基频范围）找最大峰
        lo, hi = SR // 500, SR // 50
        if hi >= len(ac):
            continue
        peak = ac[lo:hi].max()
        peak = np.clip(peak, 1e-3, 0.999)
        hnr_list.append(10 * np.log10(peak / (1 - peak)))
    return float(np.mean(hnr_list)) if hnr_list else float("nan")


METRICS = [
    ("noise_floor_db",      noise_floor_db,           "↓"),
    ("hf_noise_db_4-8k",    hf_noise_db,              "↓"),
    ("lf_rumble_db_<150",   lf_rumble_db,             "↓"),
    ("spectral_flatness",   spectral_flatness_nonspeech, "↓"),
    ("speech_rms_db",       speech_rms_db,            "—"),
    ("snr_db",              snr_db,                   "↑"),
    ("hnr_db",              hnr_db,                   "↑"),
    ("crest_db",            crest_db,                 "↑"),
]


def load(path):
    x, sr = sf.read(path, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x


def main():
    pairs = []
    for sub in sorted(os.listdir(PAIR_DIR)):
        d = os.path.join(PAIR_DIR, sub)
        a = os.path.join(d, "1_原始.wav")
        b = os.path.join(d, "2_降噪.wav")
        if os.path.exists(a) and os.path.exists(b):
            pairs.append((sub, a, b))

    print(f"样本对数: {len(pairs)}\n")

    diffs = {name: [] for name, _, _ in METRICS}
    orig_vals = {name: [] for name, _, _ in METRICS}
    new_vals = {name: [] for name, _, _ in METRICS}

    for sub, a, b in pairs:
        xo, xd = load(a), load(b)
        n = min(len(xo), len(xd))
        xo, xd = xo[:n], xd[:n]
        for name, fn, _ in METRICS:
            vo, vd = fn(xo), fn(xd)
            orig_vals[name].append(vo)
            new_vals[name].append(vd)
            diffs[name].append(vd - vo)

    # 输出汇总
    print(f"{'指标':<22} {'方向':<4} {'原始均值':>10} {'降噪均值':>10} "
          f"{'Δ均值':>10} {'Δ中位':>10} {'改善样本占比':>12}")
    print("-" * 88)

    summary = {}
    for name, _, dirn in METRICS:
        d = np.array(diffs[name])
        # 改善方向判定
        if dirn == "↓":
            improved = (d < 0).mean()
        elif dirn == "↑":
            improved = (d > 0).mean()
        else:
            improved = float("nan")

        ov = np.nanmean(orig_vals[name])
        nv = np.nanmean(new_vals[name])
        d_mean = np.nanmean(d)
        d_med = np.nanmedian(d)

        print(f"{name:<22} {dirn:<4} {ov:>10.2f} {nv:>10.2f} "
              f"{d_mean:>+10.2f} {d_med:>+10.2f} {improved*100:>10.0f}%")

        summary[name] = (d_mean, d_med, improved)

    print("-" * 88)
    print("\n说明：")
    print("  ↓ = 越低越好（噪声相关），Δ 为负代表下降")
    print("  ↑ = 越高越好（语音相关），Δ 为正代表提升")
    print("  改善样本占比 = 朝预期方向变化的样本数 / 总数")
    print(f"\n（指标说明在脚本顶部的 docstring）")


if __name__ == "__main__":
    main()
