"""
多信号目标说话人识别验证

对 output_separated_enhanced/ 的每个分离结果（轨1/轨2），算 6 个信号：
  - RMS (响度)
  - 总说话时长占比 (VAD)
  - 最长连续段长度 (VAD)
  - 首次发声时间 (onset)
  - 静音占比
  - 声纹方差 (说话人一致性反向指标)

用 ASR 字数当 ground truth（字数多 = 主说话人那一路），
看哪个信号最能正确指向主说话人。

依赖：先跑 separate_baseline.py 生成 output_separated_enhanced/
"""
import os
import sys
import glob
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEP_DIR = os.path.join(BASE_DIR, "output_separated_enhanced")
SR = 16000


# ---------- VAD ----------
def vad_energy(samples, sr=SR, frame_ms=20, threshold_db=-40):
    """
    基于能量的 VAD：>阈值 dB 视为有声。
    返回 (is_speech: bool[T_frames], frame_ms: int)
    """
    frame_size = int(sr * frame_ms / 1000)
    n_frames = len(samples) // frame_size
    if n_frames == 0:
        return np.array([], dtype=bool), frame_ms
    frames = samples[:n_frames * frame_size].reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-10)
    return db > threshold_db, frame_ms


# ---------- 信号计算 ----------
def compute_signals(samples, sr=SR):
    """对单路音频，算 6 个目标说话人识别信号"""
    is_speech, frame_ms = vad_energy(samples, sr)
    n_frames = len(is_speech)

    # 1. RMS（响度）
    rms = float(np.sqrt(np.mean(samples ** 2)))

    if n_frames == 0:
        return {
            "RMS": rms, "说话占比": 0.0, "最长段(s)": 0.0,
            "起声时间(s)": float("inf"), "静音占比": 1.0,
            "声纹方差": float("nan"),
        }

    # 2. 说话时长占比
    speech_ratio = float(is_speech.sum() / n_frames)

    # 3. 最长连续段
    max_run, cur = 0, 0
    for s in is_speech:
        if s:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    longest_sec = max_run * frame_ms / 1000.0

    # 4. 首次发声时间
    onset_indices = np.where(is_speech)[0]
    onset_sec = float(onset_indices[0] * frame_ms / 1000.0) if len(onset_indices) > 0 else float("inf")

    # 5. 静音占比
    silence_ratio = 1.0 - speech_ratio

    # 6. 声纹方差（用滑窗 speaker embedding，方差越小 → 主说话人越一致）
    speaker_var = compute_speaker_variance(samples, sr)

    return {
        "RMS": rms,
        "说话占比": speech_ratio,
        "最长段(s)": longest_sec,
        "起声时间(s)": onset_sec,
        "静音占比": silence_ratio,
        "声纹方差": speaker_var,
    }


def compute_speaker_variance(samples, sr=SR, win_sec=1.0, hop_sec=0.5):
    """
    滑窗算 speaker embedding，返回 embedding 间的余弦距离方差。
    方差小 = 整段语音都是同一人 = 主说话人轨；方差大 = 多人混杂。
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder(verbose=False)

        win = int(win_sec * sr)
        hop = int(hop_sec * sr)
        embs = []
        for start in range(0, len(samples) - win + 1, hop):
            chunk = samples[start:start + win]
            wav = preprocess_wav(chunk, source_sr=sr)
            if len(wav) < sr * 0.5:
                continue
            embs.append(enc.embed_utterance(wav))
        if len(embs) < 2:
            return 0.0

        embs = np.stack(embs)
        # 算两两余弦距离
        norm = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
        sim = norm @ norm.T
        # 上三角的距离 (1 - sim)
        K = len(embs)
        dists = []
        for i in range(K):
            for j in range(i + 1, K):
                dists.append(1.0 - sim[i, j])
        return float(np.var(dists))
    except Exception:
        return float("nan")


# ---------- ASR ground truth ----------
def get_asr_charcount(asr_model, samples):
    result = asr_model.generate(input=samples, fs=SR, batch_size_s=300, return_spk_res=True)
    text = ""
    for item in result:
        if item.get("sentence_info"):
            for sent in item["sentence_info"]:
                text += sent.get("text", "")
        else:
            text += item.get("text", "")
    # 只算中英数字
    import re
    return len(re.sub(r"[^\w一-鿿]", "", text)), text.strip()


# ---------- 主流程 ----------
def main():
    # 找 output_separated_enhanced/ 里的成对轨道
    files = sorted(glob.glob(os.path.join(SEP_DIR, "*_轨1.wav")))
    pairs = []
    for f1 in files:
        name = os.path.basename(f1).replace("_轨1.wav", "")
        f2 = os.path.join(SEP_DIR, f"{name}_轨2.wav")
        if os.path.exists(f2):
            pairs.append((name, f1, f2))

    if not pairs:
        print(f"未找到分离对，请先跑 separate_baseline.py 生成 {SEP_DIR}/")
        return

    print(f"找到 {len(pairs)} 对分离结果\n")

    print("加载 FunASR (用于 ground truth)...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", spk_model="cam++",
        disable_update=True, disable_log=True, disable_pbar=True,
    )
    print()

    # 累计每个信号的"判断准确率"
    signal_correct = defaultdict(int)
    signal_total = 0
    all_results = []

    for name, p1, p2 in pairs:
        print("=" * 90)
        print(f"📁 {name}")

        s1, _ = load_audio(p1, verbose=False)
        s2, _ = load_audio(p2, verbose=False)

        # 各路信号
        sig1 = compute_signals(s1)
        sig2 = compute_signals(s2)

        # ASR 字数当 ground truth
        n1, t1 = get_asr_charcount(asr, s1)
        n2, t2 = get_asr_charcount(asr, s2)

        if n1 == n2:
            gt = "tie"
            print(f"   ASR 字数相同（{n1}字），无法判定主说话人，跳过该样本")
        elif n1 > n2:
            gt = "track1"
        else:
            gt = "track2"
        print(f"   ASR 字数: 轨1={n1}字  轨2={n2}字  →  Ground Truth: {gt}")
        print(f"   [轨1 转写]: {t1[:60]}")
        print(f"   [轨2 转写]: {t2[:60]}")

        # 打印信号对比
        print(f"\n   {'信号':<14}{'轨1':<14}{'轨2':<14}{'指向':<10}{'是否对':<8}")
        print("   " + "-" * 60)

        # 每个信号的判断方向：True=值大者为主说话人，False=值小者为主说话人
        DIRECTION = {
            "RMS": True,
            "说话占比": True,
            "最长段(s)": True,
            "起声时间(s)": False,   # 越早发声越可能是主人
            "静音占比": False,       # 静音越少越主
            "声纹方差": False,       # 方差越小（一致）越主
        }

        if gt != "tie":
            signal_total += 1

        for k in DIRECTION:
            v1, v2 = sig1[k], sig2[k]
            if np.isnan(v1) or np.isnan(v2):
                pred, ok = "n/a", "-"
            elif DIRECTION[k]:
                pred = "track1" if v1 > v2 else "track2"
                ok = "✓" if (gt != "tie" and pred == gt) else ("✗" if gt != "tie" else "-")
            else:
                pred = "track1" if v1 < v2 else "track2"
                ok = "✓" if (gt != "tie" and pred == gt) else ("✗" if gt != "tie" else "-")

            if gt != "tie" and ok == "✓":
                signal_correct[k] += 1

            v1s = f"{v1:.4f}" if isinstance(v1, float) and not np.isnan(v1) else str(v1)
            v2s = f"{v2:.4f}" if isinstance(v2, float) and not np.isnan(v2) else str(v2)
            print(f"   {k:<14}{v1s:<14}{v2s:<14}{pred:<10}{ok:<8}")

        all_results.append((name, gt, sig1, sig2, n1, n2))
        print()

    # 汇总：每个信号的命中率
    print("=" * 90)
    print(f"📊 信号鲁棒性汇总（在 {signal_total} 个有 GT 的样本上）")
    print(f"{'信号':<14}{'命中数':<10}{'准确率':<10}")
    print("-" * 40)
    for k in ["说话占比", "最长段(s)", "起声时间(s)", "静音占比", "声纹方差", "RMS"]:
        c = signal_correct[k]
        ratio = c / signal_total if signal_total > 0 else 0
        print(f"{k:<14}{c}/{signal_total:<8}{ratio*100:.0f}%")


if __name__ == "__main__":
    main()
