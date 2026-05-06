"""
recordings_raw/ 清洗管线 (Phase 1: Layer 1 + 2 + 4)

输出:
  recordings_cleaned/
    green/    优质，直接保留（推荐用于评估/训练正样本）
    yellow/   可用，质量一般
    orange/   边缘，建议人工复核
    red/      废弃，不建议用
    report.csv      每条录音的所有指标 + tier + 原因
    summary.txt     总体统计 + 各桶 top 剔除原因

打分维度:
  Layer 1 结构: 时长、是否能解码、是否非空
  Layer 2 质量: RMS、削波、VAD 说话占比、估计 SNR
  Layer 4 ASR: 字数/秒、幻觉黑名单、重复字比例、ASR 文本
"""
import os
import sys
import csv
import glob
import re
import shutil
import time
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "recordings_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
SR = 16000

# ================ 阈值（可调）================
MIN_DURATION = 0.5    # 短于此秒数 → red
MAX_DURATION = 30.0   # 长于此秒数 → red
MIN_RMS_DB = -50.0    # 音量过小 → red
MAX_CLIP_RATIO = 0.05 # 削波样本占比 > 5% → red

MIN_VAD_RATIO_GREEN = 0.40
MIN_VAD_RATIO_YELLOW = 0.20
# < 0.20 → orange (按按钮没说话或说很少)

MIN_SNR_GREEN = 8.0
MIN_SNR_YELLOW = 3.0

MIN_CHARS_PER_SEC_GREEN = 1.5
MIN_CHARS_PER_SEC_YELLOW = 0.5

MIN_TOTAL_CHARS = 2

MAX_REPEAT_RATIO = 0.40  # 单字重复占比 > 40% → 视为低质（嗯嗯嗯/汪汪汪等）

# whisper / Paraformer 常见幻觉模式（无意义噪声上的"瞎编"输出）
HALLUCINATION_BLACKLIST = [
    "字幕", "翻译", "请不吝", "感谢您的观看", "感谢观看",
    "如果你喜欢", "订阅", "关注", "点赞",
    "中文字幕", "志愿者", "by ", "BY ",
    "明镜与点点栏目", "明镜",
]


# ================ 指标计算 ================
def vad_energy(samples, sr=SR, frame_ms=20, threshold_db=-40):
    frame_size = int(sr * frame_ms / 1000)
    n = len(samples) // frame_size
    if n == 0:
        return np.array([], dtype=bool)
    frames = samples[:n * frame_size].reshape(n, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-10)
    return db > threshold_db


def estimate_snr(samples, sr=SR):
    """用 VAD 把帧分成有声/无声，取 RMS 比当作 SNR"""
    is_speech = vad_energy(samples, sr)
    if len(is_speech) == 0:
        return float("nan")
    frame_size = int(sr * 0.02)
    n = len(is_speech)
    frames = samples[:n * frame_size].reshape(n, frame_size)
    speech_rms = np.sqrt(np.mean(frames[is_speech] ** 2)) if is_speech.any() else 0
    noise_rms = np.sqrt(np.mean(frames[~is_speech] ** 2)) if (~is_speech).any() else 1e-9
    if noise_rms < 1e-9 or speech_rms < 1e-9:
        return float("inf") if speech_rms >= 1e-9 else float("-inf")
    return 20 * np.log10(speech_rms / noise_rms)


def compute_audio_metrics(samples, sr=SR):
    """Layer 1 + 2 指标"""
    duration = len(samples) / sr
    rms = float(np.sqrt(np.mean(samples ** 2)))
    rms_db = 20 * np.log10(rms + 1e-10)
    peak = float(np.max(np.abs(samples)))
    clip_ratio = float(np.mean(np.abs(samples) > 0.99))

    is_speech = vad_energy(samples, sr)
    vad_ratio = float(is_speech.mean()) if len(is_speech) > 0 else 0.0
    snr = estimate_snr(samples, sr)

    # 最长连续段
    max_run, cur = 0, 0
    for s in is_speech:
        if s:
            cur += 1
            max_run = max(max_run, cur)
        else:
            cur = 0
    longest_speech_sec = max_run * 0.02

    return {
        "duration": duration,
        "rms_db": rms_db,
        "peak": peak,
        "clip_ratio": clip_ratio,
        "vad_ratio": vad_ratio,
        "snr_db": snr,
        "longest_speech_sec": longest_speech_sec,
    }


def normalize_text(s):
    return re.sub(r"[^\w一-鿿]", "", s)


def repeat_ratio(text):
    """单字最大重复占比，如 '嗯嗯嗯嗯啊' = 4/5 = 0.8"""
    n = normalize_text(text)
    if not n:
        return 0.0
    counter = Counter(n)
    most_common = counter.most_common(1)[0][1]
    return most_common / len(n)


def hallucination_hit(text):
    for pat in HALLUCINATION_BLACKLIST:
        if pat in text:
            return pat
    return ""


# ================ 分桶决策 ================
def classify(m, asr_text, asr_chars, asr_chars_per_sec):
    """
    根据所有指标给出 tier 和 reason。
    优先级：red 一票否决 > orange > yellow > green
    """
    reasons = []

    # === 一票否决 (red) ===
    if m.get("error"):
        return "red", f"解码失败: {m['error']}"
    if m["duration"] < MIN_DURATION:
        return "red", f"时长过短 ({m['duration']:.2f}s < {MIN_DURATION}s)"
    if m["duration"] > MAX_DURATION:
        return "red", f"时长过长 ({m['duration']:.1f}s > {MAX_DURATION}s)"
    if m["rms_db"] < MIN_RMS_DB:
        return "red", f"音量过低 ({m['rms_db']:.1f} dB)"
    if m["clip_ratio"] > MAX_CLIP_RATIO:
        return "red", f"削波严重 ({m['clip_ratio']*100:.1f}%)"
    if asr_chars < MIN_TOTAL_CHARS:
        return "red", f"ASR 字数过少 ({asr_chars} 字)"

    hit = hallucination_hit(asr_text)
    if hit:
        return "red", f"命中幻觉黑名单 '{hit}'"

    # === orange (边缘) ===
    rr = repeat_ratio(asr_text)
    # 重复字只在文本足够长时才判（避免"伟大。"这种短句被误判）
    if asr_chars >= 5 and rr > MAX_REPEAT_RATIO:
        reasons.append(f"重复字 {rr*100:.0f}%: {asr_text[:25]}")
    if m["vad_ratio"] < MIN_VAD_RATIO_YELLOW:
        reasons.append(f"VAD 占比低 ({m['vad_ratio']*100:.0f}%)")
    if not np.isnan(m["snr_db"]) and m["snr_db"] < MIN_SNR_YELLOW:
        reasons.append(f"SNR 低 ({m['snr_db']:.1f} dB)")
    if asr_chars_per_sec < MIN_CHARS_PER_SEC_YELLOW:
        reasons.append(f"语速低 ({asr_chars_per_sec:.2f} 字/秒)")

    if reasons:
        return "orange", "; ".join(reasons)

    # === green vs yellow ===
    yellow_flags = []
    if m["vad_ratio"] < MIN_VAD_RATIO_GREEN:
        yellow_flags.append(f"VAD={m['vad_ratio']*100:.0f}%")
    if not np.isnan(m["snr_db"]) and m["snr_db"] < MIN_SNR_GREEN:
        yellow_flags.append(f"SNR={m['snr_db']:.1f}dB")
    if asr_chars_per_sec < MIN_CHARS_PER_SEC_GREEN:
        yellow_flags.append(f"语速={asr_chars_per_sec:.2f}")

    if yellow_flags:
        return "yellow", "; ".join(yellow_flags)

    return "green", "全部指标达标"


# ================ 主流程 ================
def main(limit=None):
    files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.wav")))
    if limit:
        files = files[:limit]
    if not files:
        print(f"未找到音频文件: {INPUT_DIR}")
        return

    total = len(files)
    print(f"找到 {total} 个文件")
    print(f"输出目录: {OUTPUT_DIR}")

    # 准备输出目录
    for tier in ("green", "yellow", "orange", "red"):
        d = os.path.join(OUTPUT_DIR, tier)
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "report.csv")

    # 加载 ASR (FunASR Paraformer)
    print("加载 FunASR 模型...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    def transcribe(samples):
        try:
            result = asr.generate(
                input=samples, fs=SR, batch_size_s=300,
                return_spk_res=False,
            )
            text = ""
            for item in result:
                if item.get("sentence_info"):
                    for sent in item["sentence_info"]:
                        text += sent.get("text", "")
                else:
                    text += item.get("text", "")
            return text.strip()
        except Exception as e:
            return f"__ASR_ERR__:{e}"

    print("\n开始清洗...\n")
    t_start = time.time()
    rows = []
    tier_counts = Counter()
    reason_counts = Counter()

    fields = [
        "filename", "tier", "reason",
        "duration", "rms_db", "peak", "clip_ratio",
        "vad_ratio", "snr_db", "longest_speech_sec",
        "asr_chars", "asr_chars_per_sec", "repeat_ratio",
        "asr_text",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    for idx, path in enumerate(files, 1):
        fname = os.path.basename(path)
        row = {"filename": fname}

        try:
            samples, sr = load_audio(path, verbose=False)
            m = compute_audio_metrics(samples, sr)
            row.update(m)
        except Exception as e:
            row.update({
                "error": str(e),
                "duration": 0, "rms_db": -100, "peak": 0, "clip_ratio": 0,
                "vad_ratio": 0, "snr_db": float("nan"), "longest_speech_sec": 0,
            })
            tier, reason = "red", f"解码失败: {e}"
            row["tier"] = tier
            row["reason"] = reason
            row["asr_chars"] = 0
            row["asr_chars_per_sec"] = 0
            row["repeat_ratio"] = 0
            row["asr_text"] = ""
            tier_counts[tier] += 1
            reason_counts[reason] += 1
            writer.writerow(_filter_row(row, fields))
            try:
                shutil.copy2(path, os.path.join(OUTPUT_DIR, tier, fname))
            except Exception:
                pass
            _print_progress(idx, total, t_start, tier_counts)
            continue

        # 如果连基础质量都不行，可以跳过 ASR 节省时间
        skip_asr = False
        if m["duration"] < MIN_DURATION or m["rms_db"] < MIN_RMS_DB:
            skip_asr = True

        if skip_asr:
            asr_text, asr_chars = "", 0
            asr_chars_per_sec = 0.0
        else:
            asr_text = transcribe(samples)
            if asr_text.startswith("__ASR_ERR__"):
                asr_text = ""
            asr_chars = len(normalize_text(asr_text))
            asr_chars_per_sec = asr_chars / max(m["duration"], 0.1)

        rr = repeat_ratio(asr_text)
        tier, reason = classify(m, asr_text, asr_chars, asr_chars_per_sec)

        row.update({
            "tier": tier, "reason": reason,
            "asr_chars": asr_chars,
            "asr_chars_per_sec": round(asr_chars_per_sec, 3),
            "repeat_ratio": round(rr, 3),
            "asr_text": asr_text,
        })
        writer.writerow(_filter_row(row, fields))

        # 复制到对应桶
        try:
            shutil.copy2(path, os.path.join(OUTPUT_DIR, tier, fname))
        except Exception:
            pass

        tier_counts[tier] += 1
        reason_counts[reason] += 1

        # 定期 flush
        if idx % 50 == 0:
            csv_file.flush()
            _print_progress(idx, total, t_start, tier_counts)

    csv_file.close()

    # === 总结 ===
    elapsed = time.time() - t_start
    print(f"\n\n{'='*78}")
    print(f"✓ 清洗完成，总耗时 {elapsed/60:.1f} 分钟")
    print(f"\n📊 分桶统计 (共 {total} 条):")
    for tier in ("green", "yellow", "orange", "red"):
        c = tier_counts[tier]
        print(f"  {tier:<8} {c:>5}  ({c/total*100:5.1f}%)")

    print(f"\n📋 Top 10 进入桶的原因:")
    for reason, count in reason_counts.most_common(10):
        print(f"  {count:>4}  {reason}")

    print(f"\n输出位置: {OUTPUT_DIR}/")
    print(f"  - 各 tier 子目录已复制对应文件")
    print(f"  - report.csv  (按 tier 列排序后能直接看清洗结果)")

    # 写 summary.txt
    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"清洗时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总文件数: {total}\n")
        f.write(f"耗时: {elapsed/60:.1f} 分钟\n\n")
        f.write("分桶统计:\n")
        for tier in ("green", "yellow", "orange", "red"):
            c = tier_counts[tier]
            f.write(f"  {tier}: {c} ({c/total*100:.1f}%)\n")
        f.write(f"\nTop 20 原因:\n")
        for reason, count in reason_counts.most_common(20):
            f.write(f"  {count:>4}  {reason}\n")
    print(f"  - summary.txt")


def _filter_row(row, fields):
    return {k: row.get(k, "") for k in fields}


def _print_progress(idx, total, t_start, tier_counts):
    elapsed = time.time() - t_start
    rate = idx / elapsed if elapsed > 0 else 0
    remaining = (total - idx) / rate if rate > 0 else 0
    g = tier_counts["green"]
    y = tier_counts["yellow"]
    o = tier_counts["orange"]
    r = tier_counts["red"]
    print(f"  [{idx:>4}/{total}] {rate:.1f} 条/秒 剩余 {remaining/60:.1f} 分钟 | "
          f"🟢{g} 🟡{y} 🟠{o} 🔴{r}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件（用于试跑）")
    p.add_argument("--out", type=str, default=None, help="覆盖输出目录")
    args = p.parse_args()
    if args.out:
        OUTPUT_DIR = args.out
    main(limit=args.limit)
