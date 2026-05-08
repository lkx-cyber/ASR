"""
横向对比所有「音频处理方案」在 3 个场景下提取主说话人的能力。

场景：
  S1 单人clean  ── 5 条 from green++ (有 ground truth, 算 CER)
  S2 多人        ── 5 条 from verify_multi/ (无 GT, 看 ASR 文本质量 + 分离两路相似度)
  S3 嘈杂        ── 5 条 from yellow tier 中 SNR 最低的 (无 GT, 看 SNR/ASR 改善)

方法（音频处理边界内，不含 ASR 调参）：
  M0  raw                       不处理（基线）
  M1  dfn3                      DFN3 50%干湿混合降噪
  M2  pad200                    开头加 200ms 静音
  M3  dfn3+pad200              组合
  M4  baseline_sep              ONNX 分离 + Wiener，自动取主说话人轨
  M5  pad200+baseline_sep       padding + 分离

每个方法都会输出"目标说话人"音频，所有方案在输出格式上是同质的，可直接对比。

输出 test_audio_methods/<timestamp>/:
  summary.txt     场景×方法 总览（含决策建议）
  detail.csv      每条样本所有指标
  wavs/<scenario>/<filename>/M*.wav   每个方法处理后的 wav，供听感校验
"""
import argparse
import csv
import os
import random
import sys
import time
import warnings
from datetime import datetime

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ============== 工具 ==============
def load_audio(path, target_sr=16000):
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        import librosa
        x, sr = librosa.load(path, sr=None, mono=True)
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return x, target_sr


def save_audio(path, x, sr=16000):
    sf.write(path, x.astype(np.float32), sr)


def db(p, eps=1e-12):
    return 10 * np.log10(max(p, eps))


def compute_audio_metrics(x):
    """音频客观指标：噪声底 / SNR / 高频噪声"""
    frame = 400  # 25ms @ 16k
    if len(x) < frame:
        return {"noise_floor_db": 0, "speech_rms_db": 0, "snr_db": 0, "hf_noise_db": 0}
    n_frames = len(x) // frame
    powers = np.array([np.mean(x[i*frame:(i+1)*frame]**2) for i in range(n_frames)])
    powers_sorted = np.sort(powers)
    noise_floor = db(powers_sorted[: max(1, n_frames // 10)].mean())
    speech_rms = db(powers_sorted[-max(1, n_frames // 2):].mean())
    snr = speech_rms - noise_floor
    # 高频段
    fft = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1/16000)
    hf = np.mean(np.abs(fft[(freqs >= 4000) & (freqs <= 8000)]) ** 2) / max(1, len(x))
    return {
        "noise_floor_db": float(noise_floor),
        "speech_rms_db": float(speech_rms),
        "snr_db": float(snr),
        "hf_noise_db": float(db(hf)),
    }


def normalize_text(s):
    """归一化用于 CER"""
    if not s:
        return ""
    try:
        from opencc import OpenCC
        s = OpenCC("t2s").convert(s)
    except Exception:
        pass
    import string
    PUNCT = set("，。！？、；：""''（）【】《》「」『』〈〉…—·～￥" + string.punctuation + " \t\n\r")
    out = []
    for ch in s:
        c = ord(ch)
        if c == 0x3000: ch = " "
        elif 0xFF01 <= c <= 0xFF5E: ch = chr(c - 0xFEE0)
        out.append(ch)
    return "".join(ch for ch in "".join(out).lower() if ch not in PUNCT)


def edit_distance(s1, s2):
    if s1 == s2: return 0
    m, n = len(s1), len(s2)
    if m == 0 or n == 0: return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0]*n
        for j in range(1, n + 1):
            cur[j] = prev[j-1] if s1[i-1]==s2[j-1] else 1+min(prev[j], cur[j-1], prev[j-1])
        prev = cur
    return prev[n]


def cer(ref, hyp):
    r, h = normalize_text(ref), normalize_text(hyp)
    return edit_distance(r, h), len(r) or 1


def speaker_cosine(a, b, sr=16000):
    """两段音频的说话人余弦相似度（越低越像两个不同人）"""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder(verbose=False)
        e1 = enc.embed_utterance(preprocess_wav(a, source_sr=sr))
        e2 = enc.embed_utterance(preprocess_wav(b, source_sr=sr))
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9))
    except Exception as e:
        return float("nan")


# ============== 方法：每个返回"主说话人音频" + 可选副轨 ==============
METHODS = {}


def register(name):
    def deco(fn):
        METHODS[name] = fn
        return fn
    return deco


def pad_silence(audio, sr, ms=200):
    pad = np.zeros(int(ms / 1000 * sr), dtype=np.float32)
    return np.concatenate([pad, audio])


# ============== 抽样 ==============
def pick_samples(seed=42):
    """各场景挑 5 条"""
    random.seed(seed)
    samples = {}

    # S1 单人clean: from groundtruth.csv
    gt_path = os.path.join(BASE_DIR, "groundtruth.csv")
    with open(gt_path, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)
                if r["correct_text"].strip() and r["correct_text"].strip() != "/n"]
    s1_samples = random.sample(rows, 5)
    samples["S1_单人clean"] = [
        (r["filename"],
         os.path.join(BASE_DIR, "recordings_cleaned_v2/green++", r["filename"]),
         r["correct_text"])
        for r in s1_samples
    ]

    # S2 多人: from verify_multi/
    multi_dir = os.path.join(BASE_DIR, "verify_multi")
    multi_files = sorted(os.listdir(multi_dir))
    s2_picked = random.sample(multi_files, 5)
    samples["S2_多人"] = [
        (f, os.path.join(multi_dir, f), None) for f in s2_picked
    ]

    # S3 嘈杂: yellow tier 中 SNR 最低 + 有内容（避免空音频）
    rep_path = os.path.join(BASE_DIR, "recordings_cleaned_v2/report_v2.csv")
    with open(rep_path, encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    noisy = [r for r in all_rows if r["tier"] == "yellow"
             and float(r.get("snr_db") or 0) > 5
             and int(r.get("asr_chars") or 0) > 3]
    noisy.sort(key=lambda r: float(r["snr_db"]))
    # 挑前 30 中随机 5 条（避免集中在某段）
    pool = noisy[:30]
    s3_picked = random.sample(pool, 5)
    yellow_dir = os.path.join(BASE_DIR, "recordings_cleaned_v2/yellow")
    samples["S3_嘈杂"] = [
        (r["filename"], os.path.join(yellow_dir, r["filename"]), None)
        for r in s3_picked
    ]

    # S4 自录多人: samples_test/ (16kHz 原生，非 8k 上采，最干净测试源)
    samples_test_dir = os.path.join(BASE_DIR, "samples_test")
    if os.path.isdir(samples_test_dir):
        files = sorted([f for f in os.listdir(samples_test_dir)
                        if f.endswith((".wav", ".m4a", ".mp3"))])
        samples["S4_自录多人"] = [
            (f, os.path.join(samples_test_dir, f), None) for f in files
        ]

    return samples


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-root", default="test_audio_methods")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, args.out_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)
    wav_dir = os.path.join(out_dir, "wavs")

    samples = pick_samples(args.seed)
    print(f"📋 共 {sum(len(v) for v in samples.values())} 条样本，覆盖 {len(samples)} 个场景\n")

    # 加载模型（按需）
    print("加载模型...")
    import onnxruntime as ort
    from separate_baseline import separate_one as baseline_sep
    sep_session = ort.InferenceSession(
        os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.onnx"))
    )
    from denoise import Denoiser
    dn = Denoiser()

    # FRCRN（ClearerVoice 中文场景降噪），可能不可用 → 容错
    frcrn = None
    try:
        from frcrn_denoise import FRCRNDenoiser
        frcrn = FRCRNDenoiser()
        # 预热（避免首次推理慢拉低 RTF 数据）
        frcrn(np.zeros(16000, dtype=np.float32))
        print("  ✓ FRCRN 加载完成")
    except Exception as e:
        print(f"  ⚠️ FRCRN 不可用，跳过 M1b: {e}")

    from funasr import AutoModel
    fa = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    def fa_text(x):
        try:
            r = fa.generate(input=x, fs=16000, batch_size_s=300, disable_pbar=True)
            return r[0].get("text", "").strip() if r else ""
        except Exception as e:
            return f"<err:{e}>"

    print("  ✓\n")

    # 注册方法（在模型加载后）
    @register("M0_raw")
    def m0(audio, sr):
        return audio, None  # main, sub

    @register("M1_dfn3")
    def m1(audio, sr):
        return dn(audio), None

    if frcrn is not None:
        @register("M1b_frcrn")
        def m1b(audio, sr):
            return frcrn(audio), None

        @register("M3b_frcrn+pad200")
        def m3b(audio, sr):
            return pad_silence(frcrn(audio), sr, 200), None

    @register("M2_pad200")
    def m2(audio, sr):
        return pad_silence(audio, sr, 200), None

    @register("M3_dfn3+pad200")
    def m3(audio, sr):
        return pad_silence(dn(audio), sr, 200), None

    @register("M4_baseline_sep")
    def m4(audio, sr):
        tracks = baseline_sep(sep_session, audio)
        # 按能量排序，主轨 = 能量最大
        idx_sorted = sorted(range(len(tracks)),
                           key=lambda i: -np.mean(tracks[i] ** 2))
        return tracks[idx_sorted[0]], tracks[idx_sorted[1]] if len(tracks) > 1 else None

    @register("M5_pad+baseline_sep")
    def m5(audio, sr):
        padded = pad_silence(audio, sr, 200)
        tracks = baseline_sep(sep_session, padded)
        idx_sorted = sorted(range(len(tracks)),
                           key=lambda i: -np.mean(tracks[i] ** 2))
        return tracks[idx_sorted[0]], tracks[idx_sorted[1]] if len(tracks) > 1 else None

    if frcrn is not None:
        @register("M6_frcrn+sep")
        def m6(audio, sr):
            """FRCRN 降噪 → baseline 分离（取主轨）"""
            denoised = frcrn(audio)
            tracks = baseline_sep(sep_session, denoised)
            idx_sorted = sorted(range(len(tracks)), key=lambda i: -np.mean(tracks[i] ** 2))
            return tracks[idx_sorted[0]], tracks[idx_sorted[1]] if len(tracks) > 1 else None

        @register("M7_frcrn+pad+sep")
        def m7(audio, sr):
            """全配置：FRCRN → 200ms padding → baseline 分离（取主轨）"""
            denoised = frcrn(audio)
            padded = pad_silence(denoised, sr, 200)
            tracks = baseline_sep(sep_session, padded)
            idx_sorted = sorted(range(len(tracks)), key=lambda i: -np.mean(tracks[i] ** 2))
            return tracks[idx_sorted[0]], tracks[idx_sorted[1]] if len(tracks) > 1 else None

    method_names = list(METHODS.keys())
    print(f"📋 方法: {method_names}\n")

    # ============== 跑测试 ==============
    # results: list of dict, 每条样本一行
    results = []

    for scenario, items in samples.items():
        print(f"\n{'=' * 80}")
        print(f"【{scenario}】 {len(items)} 条")
        print(f"{'=' * 80}")
        for sample_idx, (fname, path, gt) in enumerate(items, 1):
            print(f"\n  [{sample_idx}/{len(items)}] {fname[:14]}")
            if not os.path.exists(path):
                print(f"    ⚠️ 文件不存在，跳过")
                continue
            audio, sr = load_audio(path)
            sample_wav_dir = os.path.join(wav_dir, scenario, fname.replace(".wav", "")[:14])
            os.makedirs(sample_wav_dir, exist_ok=True)
            save_audio(os.path.join(sample_wav_dir, "00_原始.wav"), audio)
            if gt:
                with open(os.path.join(sample_wav_dir, "gt.txt"), "w", encoding="utf-8") as f:
                    f.write(gt)

            row = {
                "scenario": scenario, "filename": fname, "duration": len(audio)/sr,
                "ground_truth": gt or "",
            }

            # 跑每个方法
            for m_name in method_names:
                t0 = time.time()
                main_track, sub_track = METHODS[m_name](audio, sr)
                proc_secs = time.time() - t0

                # 保存主轨
                save_audio(os.path.join(sample_wav_dir, f"{m_name}_主.wav"), main_track)
                if sub_track is not None:
                    save_audio(os.path.join(sample_wav_dir, f"{m_name}_副.wav"), sub_track)

                # 音频指标（主轨）
                metrics = compute_audio_metrics(main_track)

                # ASR
                t1 = time.time()
                asr_text = fa_text(main_track)
                asr_secs = time.time() - t1

                # CER (S1 only)
                if gt:
                    ed, ref_len = cer(gt, asr_text)
                    cer_pct = 100 * ed / ref_len
                else:
                    cer_pct = None

                # 说话人分离度（仅 sep 方法）
                spk_sim = None
                if sub_track is not None:
                    spk_sim = speaker_cosine(main_track, sub_track, sr)

                row[f"{m_name}_proc_secs"] = round(proc_secs, 2)
                row[f"{m_name}_asr_text"] = asr_text
                row[f"{m_name}_asr_chars"] = len(asr_text)
                row[f"{m_name}_snr_db"] = round(metrics["snr_db"], 2)
                row[f"{m_name}_noise_floor_db"] = round(metrics["noise_floor_db"], 2)
                row[f"{m_name}_hf_noise_db"] = round(metrics["hf_noise_db"], 2)
                if cer_pct is not None:
                    row[f"{m_name}_cer"] = round(cer_pct, 2)
                if spk_sim is not None and not np.isnan(spk_sim):
                    row[f"{m_name}_spk_sim"] = round(spk_sim, 3)

                # 控制台简报
                tag = f"cer={cer_pct:.0f}%" if cer_pct is not None \
                      else f"chars={len(asr_text)}"
                print(f"    {m_name:<22} {tag}  snr={metrics['snr_db']:.1f}dB  asr: {asr_text[:35]}")

            results.append(row)

    # ============== 输出 detail.csv ==============
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        # 列排序：基础列 + 按方法分组
        base_cols = ["scenario", "filename", "duration", "ground_truth"]
        method_cols = []
        for m in method_names:
            for suffix in ["asr_text", "asr_chars", "cer", "spk_sim",
                           "snr_db", "noise_floor_db", "hf_noise_db", "proc_secs"]:
                key = f"{m}_{suffix}"
                if key in all_keys:
                    method_cols.append(key)
        ordered_cols = base_cols + method_cols

        detail_path = os.path.join(out_dir, "detail.csv")
        with open(detail_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered_cols)
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in ordered_cols})

    # ============== 输出 summary.txt ==============
    lines = []
    lines.append("=" * 80)
    lines.append(f"音频处理方案横向对比  {timestamp}")
    lines.append("=" * 80)
    lines.append(f"\n方法: {method_names}\n")

    # 按场景汇总
    for scenario in samples:
        scenario_rows = [r for r in results if r["scenario"] == scenario]
        if not scenario_rows:
            continue
        lines.append("=" * 80)
        lines.append(f"【{scenario}】 {len(scenario_rows)} 条")
        lines.append("=" * 80)

        # 汇总表
        lines.append("")
        if scenario == "S1_单人clean":
            lines.append(f"{'方法':<24} {'平均CER':>10} {'整体CER':>10} {'平均SNR':>10} {'平均ASR字':>10}")
        else:
            lines.append(f"{'方法':<24} {'平均SNR':>10} {'平均HF噪声':>12} {'平均ASR字':>10}")
        lines.append("-" * 80)

        for m_name in method_names:
            cer_vals = [r.get(f"{m_name}_cer") for r in scenario_rows
                       if r.get(f"{m_name}_cer") is not None]
            snr_vals = [r.get(f"{m_name}_snr_db", 0) for r in scenario_rows]
            hf_vals = [r.get(f"{m_name}_hf_noise_db", 0) for r in scenario_rows]
            chars_vals = [r.get(f"{m_name}_asr_chars", 0) for r in scenario_rows]
            sim_vals = [r.get(f"{m_name}_spk_sim") for r in scenario_rows
                       if r.get(f"{m_name}_spk_sim") is not None]

            if scenario == "S1_单人clean" and cer_vals:
                # 整体 CER：用 ed 加权
                eds = []
                ref_lens = []
                for r in scenario_rows:
                    text = r.get(f"{m_name}_asr_text", "")
                    gt = r.get("ground_truth", "")
                    if gt:
                        e, l = cer(gt, text)
                        eds.append(e); ref_lens.append(l)
                overall = 100 * sum(eds) / max(1, sum(ref_lens))
                avg_cer = float(np.mean(cer_vals))
                lines.append(f"{m_name:<24} {avg_cer:>9.2f}% {overall:>9.2f}% "
                            f"{np.mean(snr_vals):>9.2f}dB {np.mean(chars_vals):>10.1f}")
            else:
                lines.append(f"{m_name:<24} {np.mean(snr_vals):>9.2f}dB "
                            f"{np.mean(hf_vals):>11.2f}dB {np.mean(chars_vals):>10.1f}")

            # 说话人分离度（仅 sep 方法 + S2 多人最有意义）
            if sim_vals and scenario == "S2_多人":
                lines.append(f"  ↳ 说话人区分度（主轨 vs 副轨 余弦相似度）平均: {np.mean(sim_vals):.3f} "
                            f"(<0.7=分得开)")

        # 难 case
        lines.append("\n样本明细：")
        for r in scenario_rows:
            lines.append(f"  📁 {r['filename'][:14]}  ({r['duration']:.1f}s)")
            if r.get("ground_truth"):
                lines.append(f"     [真值]  {r['ground_truth']}")
            for m_name in method_names:
                txt = r.get(f"{m_name}_asr_text", "")
                cer_v = r.get(f"{m_name}_cer")
                tag = f"cer={cer_v:.0f}%" if cer_v is not None else f"{len(txt)}字"
                lines.append(f"     {m_name:<22} {tag:<8} {txt[:50]}")
        lines.append("")

    # 决策建议
    lines.append("=" * 80)
    lines.append("【决策依据】")
    lines.append("=" * 80)
    lines.append("""
- S1 单人clean：直接看 CER，越低越好
- S2 多人：看 ASR 文本质量 + 说话人区分度（M4/M5 的副轨是否分得开）
- S3 嘈杂：看 SNR 改善 + ASR 字数（字数过低=信号被压死，字数稳定=干净处理）

人耳校验：进入 wavs/<scenario>/<filename>/ 听 00_原始.wav 和 M*_主.wav 对比
""")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print()
    print("\n".join(lines[-30:]))
    print()
    print(f"✓ 详情 CSV: {detail_path}")
    print(f"✓ 总结: {summary_path}")
    print(f"✓ 中间 wav (供听感校验): {wav_dir}")


if __name__ == "__main__":
    main()
