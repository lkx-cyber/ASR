"""
clean_dataset_v2.py - 增强版数据集清洗

在 v1 (clean_dataset.py) 基础上加 5 个新信号，针对性筛掉:
  - 误触录音 (没真说话, 只是环境噪声)
  - 人耳听不懂的低质音频
  - 远场弱信号 (孩子离麦太远)
  - ASR 瞎编但通过基础指标的样本

5 个新信号:
  1. Silero VAD 真实语音占比          → 筛掉非语音
  2. 谱熵 (spectral entropy)           → 筛掉噪声主导段
  3. F0 voiced 占比                    → 验证真有人声
  4. 双引擎 ASR 拼音 CER 一致性         → 筛"模型瞎猜"案例
  5. RMS 动态范围                      → 筛"平噪声/无说话"

新分桶:
  green++ (premium):  原 green + 5 个新信号全过
  green+:             原 green + 5 个新信号过 3-4 个
  green:              原 green + 5 个新信号过 ≤2 个
  yellow/orange/red:  原分类不变

输出:
  recordings_cleaned_v2/
    green++/  green+/  green/  yellow/  orange/  red/
    report_v2.csv  全部 18 个指标 + 双 ASR 文本 + 分桶 + 原因
    summary_v2.txt
"""
import argparse
import csv
import glob
import os
import re
import shutil
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio
from clean_dataset import (
    compute_audio_metrics, normalize_text, repeat_ratio, hallucination_hit,
    HALLUCINATION_BLACKLIST,
    MIN_DURATION, MAX_DURATION, MIN_RMS_DB, MAX_CLIP_RATIO, MIN_TOTAL_CHARS,
    MIN_VAD_RATIO_YELLOW, MIN_VAD_RATIO_GREEN, MIN_SNR_YELLOW, MIN_SNR_GREEN,
    MIN_CHARS_PER_SEC_YELLOW, MIN_CHARS_PER_SEC_GREEN, MAX_REPEAT_RATIO,
)
from device_utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "recordings_raw")
OUTPUT_DIR = os.path.join(BASE_DIR, "recordings_cleaned_v2")
SR = 16000

# ============== 5 个新信号的阈值 ==============
SILERO_SPEECH_RATIO_OK = 0.30   # 真实语音占比下限
SPECTRAL_ENTROPY_MAX = 0.85     # 谱熵上限 (越低越像结构化语音)
F0_VOICED_RATIO_OK = 0.15       # F0 有声帧占比下限
DUAL_ASR_CER_MAX = 0.40         # 双引擎拼音 CER 上限
RMS_DYNAMICS_MIN = 0.20         # log RMS 标准差下限 (太平的多半是噪声)

# 通过几个新信号判级
N_PREMIUM_PASS = 5      # 全过 → green++
N_GOOD_PASS = 3         # ≥3 过 → green+


# ============== 5 个新信号实现 ==============
_silero_model = None
_silero_utils = None

def _load_silero():
    global _silero_model, _silero_utils
    if _silero_model is None:
        import torch
        torch.set_num_threads(1)
        try:
            _silero_model, _silero_utils = torch.hub.load(
                "snakers4/silero-vad", "silero_vad",
                trust_repo=True, verbose=False,
            )
        except Exception:
            # 离线 fallback：用 pip 安装的 silero-vad
            from silero_vad import load_silero_vad, get_speech_timestamps
            _silero_model = load_silero_vad()
            _silero_utils = (get_speech_timestamps,)
    return _silero_model, _silero_utils


def silero_speech_ratio(audio, sr=SR):
    """Silero VAD 真实语音 / 总时长"""
    import torch
    model, utils = _load_silero()
    get_speech_timestamps = utils[0]
    audio_t = torch.from_numpy(audio.astype(np.float32))
    try:
        timestamps = get_speech_timestamps(audio_t, model, sampling_rate=sr)
    except Exception:
        return float("nan")
    total = sum(t["end"] - t["start"] for t in timestamps)
    return total / len(audio) if len(audio) > 0 else 0.0


def spectral_entropy(audio, n_fft=512, hop=256):
    """谱熵: 噪声 ≈ 1, 结构化语音较低"""
    n = len(audio)
    if n < n_fft:
        return float("nan")
    win = np.hanning(n_fft)
    n_frames = (n - n_fft) // hop + 1
    if n_frames < 3:
        return float("nan")
    H_list = []
    for i in range(n_frames):
        frame = audio[i*hop : i*hop + n_fft] * win
        spec = np.abs(np.fft.rfft(frame)) ** 2 + 1e-12
        p = spec / spec.sum()
        H = -np.sum(p * np.log(p))
        H_list.append(H / np.log(len(p)))  # normalize 0-1
    return float(np.mean(H_list))


def f0_voiced_ratio(audio, sr=SR):
    """F0 有声帧占比 (用 librosa.pyin)"""
    try:
        import librosa
        f0, voiced, _ = librosa.pyin(
            audio.astype(np.float32),
            fmin=80, fmax=600, sr=sr,
            frame_length=1024, hop_length=256,
        )
        return float(np.nanmean(voiced.astype(np.float32))) if voiced is not None else 0.0
    except Exception:
        return float("nan")


def rms_dynamics(audio, sr=SR, frame_ms=20):
    """log RMS 标准差: 大 = 有 peak/valley 像说话, 小 = 平噪声"""
    frame_n = int(sr * frame_ms / 1000)
    if len(audio) < frame_n * 5:
        return float("nan")
    n_frames = len(audio) // frame_n
    frames = audio[:n_frames * frame_n].reshape(n_frames, frame_n)
    rms = np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10
    log_rms = np.log10(rms)
    return float(np.std(log_rms))


# ----- 双引擎 ASR 一致性 -----
_pinyin_cache = None

def _to_pinyin(text):
    global _pinyin_cache
    if _pinyin_cache is None:
        try:
            from pypinyin import lazy_pinyin
            _pinyin_cache = lazy_pinyin
        except ImportError:
            return None
    norm = normalize_text(text)
    if not norm:
        return ""
    return "".join(_pinyin_cache(norm))


def _edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m]


def pinyin_cer(text_a, text_b):
    """两段中文文本的拼音 CER (基于字母编辑距离)"""
    pa, pb = _to_pinyin(text_a), _to_pinyin(text_b)
    if pa is None or pb is None:
        return float("nan")
    if not pa and not pb:
        return 0.0
    if not pa or not pb:
        return 1.0
    return _edit_distance(pa, pb) / max(len(pa), len(pb))


# ============== 分桶决策 ==============
def classify_v2(m, asr_text, asr_chars, asr_chars_per_sec,
                silero_ratio, spec_ent, f0_ratio, dual_cer, rms_dyn):
    """
    返回 (tier, reason, n_new_passed, new_passed_list)
    """
    # === red 一票否决 (原有逻辑) ===
    if m.get("error"):
        return "red", f"解码失败: {m['error']}", 0, []
    if m["duration"] < MIN_DURATION:
        return "red", f"时长过短 ({m['duration']:.2f}s)", 0, []
    if m["duration"] > MAX_DURATION:
        return "red", f"时长过长 ({m['duration']:.1f}s)", 0, []
    if m["rms_db"] < MIN_RMS_DB:
        return "red", f"音量过低 ({m['rms_db']:.1f}dB)", 0, []
    if m["clip_ratio"] > MAX_CLIP_RATIO:
        return "red", f"削波 ({m['clip_ratio']*100:.1f}%)", 0, []
    if asr_chars < MIN_TOTAL_CHARS:
        return "red", f"ASR 字数过少 ({asr_chars})", 0, []
    hit = hallucination_hit(asr_text)
    if hit:
        return "red", f"幻觉黑名单 '{hit}'", 0, []

    # === orange/yellow (原有逻辑) ===
    soft_reasons = []
    rr = repeat_ratio(asr_text)
    if asr_chars >= 5 and rr > MAX_REPEAT_RATIO:
        soft_reasons.append(f"重复字 {rr*100:.0f}%")
    if m["vad_ratio"] < MIN_VAD_RATIO_YELLOW:
        soft_reasons.append(f"VAD={m['vad_ratio']*100:.0f}%")
    if not np.isnan(m["snr_db"]) and m["snr_db"] < MIN_SNR_YELLOW:
        soft_reasons.append(f"SNR={m['snr_db']:.1f}dB")
    if asr_chars_per_sec < MIN_CHARS_PER_SEC_YELLOW:
        soft_reasons.append(f"语速={asr_chars_per_sec:.2f}")
    if soft_reasons:
        return "orange", "; ".join(soft_reasons), 0, []

    # === yellow (原 green 失分) ===
    yellow_flags = []
    if m["vad_ratio"] < MIN_VAD_RATIO_GREEN:
        yellow_flags.append(f"VAD={m['vad_ratio']*100:.0f}%")
    if not np.isnan(m["snr_db"]) and m["snr_db"] < MIN_SNR_GREEN:
        yellow_flags.append(f"SNR={m['snr_db']:.1f}dB")
    if asr_chars_per_sec < MIN_CHARS_PER_SEC_GREEN:
        yellow_flags.append(f"语速={asr_chars_per_sec:.2f}")
    if yellow_flags:
        return "yellow", "; ".join(yellow_flags), 0, []

    # === 现在过了原版 green，看 5 个新信号 ===
    new_passed = []
    new_failed = []

    def _check(name, value, cmp, threshold):
        if np.isnan(value):
            new_failed.append(f"{name}=NaN")
            return False
        if cmp == "ge":
            ok = value >= threshold
        elif cmp == "le":
            ok = value <= threshold
        else:
            ok = False
        if ok:
            new_passed.append(name)
        else:
            new_failed.append(f"{name}={value:.2f}")
        return ok

    _check("silero", silero_ratio, "ge", SILERO_SPEECH_RATIO_OK)
    _check("spec_ent", spec_ent, "le", SPECTRAL_ENTROPY_MAX)
    _check("f0", f0_ratio, "ge", F0_VOICED_RATIO_OK)
    _check("dual_cer", dual_cer, "le", DUAL_ASR_CER_MAX)
    _check("rms_dyn", rms_dyn, "ge", RMS_DYNAMICS_MIN)

    n_pass = len(new_passed)
    if n_pass == N_PREMIUM_PASS:
        return "green++", f"5/5 新信号全过", n_pass, new_passed
    elif n_pass >= N_GOOD_PASS:
        return "green+", f"{n_pass}/5 新信号过: {','.join(new_passed)}", n_pass, new_passed
    else:
        msg = f"仅 {n_pass}/5 新信号过"
        if new_failed:
            msg += f"; 失败: {','.join(new_failed[:3])}"
        return "green", msg, n_pass, new_passed


# ============== 主流程 ==============
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", default=INPUT_DIR)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--no-copy", action="store_true",
                        help="不复制 wav 到分桶目录（只生成 csv）")
    parser.add_argument("--skip-sensevoice", action="store_true",
                        help="跳过 SenseVoice (仅用 Paraformer，dual_cer 信号失效)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.input, "*.wav")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"没找到 wav: {args.input}")
        return
    total = len(files)
    print(f"将处理 {total} 个文件")

    # 输出
    tiers = ("green++", "green+", "green", "yellow", "orange", "red")
    for t in tiers:
        os.makedirs(os.path.join(args.output, t), exist_ok=True)
    csv_path = os.path.join(args.output, "report_v2.csv")

    # 加载模型
    device = get_device()
    print(f"[device] {device}")

    print("加载 FunASR Paraformer...")
    from funasr import AutoModel
    asr_pf = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
        device=device,
    )

    asr_sv = None
    if not args.skip_sensevoice:
        print("加载 SenseVoice (双引擎一致性需要)...")
        try:
            asr_sv = AutoModel(
                model="iic/SenseVoiceSmall",
                disable_update=True, disable_log=True, disable_pbar=True,
                device=device,
            )
        except Exception as e:
            print(f"  SenseVoice 加载失败 ({e})，dual_cer 信号将失效")

    print("预热 Silero VAD...")
    _load_silero()

    print("\n开始处理...\n")

    def _transcribe(model, samples):
        try:
            r = model.generate(input=samples, fs=SR, batch_size_s=300)
            text = ""
            for item in r:
                if item.get("sentence_info"):
                    for sent in item["sentence_info"]:
                        text += sent.get("text", "")
                else:
                    text += item.get("text", "")
            return text.strip()
        except Exception:
            return ""

    fields = [
        "filename", "tier", "reason", "n_new_passed", "new_passed_list",
        # 原版 8 项
        "duration", "rms_db", "peak", "clip_ratio",
        "vad_ratio", "snr_db", "longest_speech_sec",
        # 原版 ASR 4 项
        "asr_text_pf", "asr_chars", "asr_chars_per_sec", "repeat_ratio",
        # 5 个新信号
        "silero_speech_ratio", "spectral_entropy",
        "f0_voiced_ratio", "rms_dynamics", "dual_pinyin_cer",
        # 第二引擎 ASR 文本（参考）
        "asr_text_sv",
    ]

    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    tier_counts = Counter()
    new_pass_dist = Counter()
    t_start = time.time()

    for idx, path in enumerate(files, 1):
        fname = os.path.basename(path)
        row = {"filename": fname}

        # 加载
        try:
            samples, sr = load_audio(path, verbose=False)
        except Exception as e:
            row.update({
                "tier": "red", "reason": f"解码失败: {e}",
                "n_new_passed": 0, "new_passed_list": "",
                "duration": 0, "rms_db": -100, "peak": 0, "clip_ratio": 0,
                "vad_ratio": 0, "snr_db": float("nan"), "longest_speech_sec": 0,
                "asr_text_pf": "", "asr_chars": 0, "asr_chars_per_sec": 0, "repeat_ratio": 0,
                "silero_speech_ratio": 0, "spectral_entropy": float("nan"),
                "f0_voiced_ratio": 0, "rms_dynamics": float("nan"), "dual_pinyin_cer": float("nan"),
                "asr_text_sv": "",
            })
            writer.writerow(row)
            tier_counts["red"] += 1
            try:
                shutil.copy2(path, os.path.join(args.output, "red", fname))
            except Exception:
                pass
            continue

        # 原版 8 项
        m = compute_audio_metrics(samples, sr)
        row.update(m)

        # ASR (Paraformer) - 必须
        skip_asr = m["duration"] < MIN_DURATION or m["rms_db"] < MIN_RMS_DB
        if skip_asr:
            asr_text_pf = ""
        else:
            asr_text_pf = _transcribe(asr_pf, samples)
        asr_chars = len(normalize_text(asr_text_pf))
        asr_chars_per_sec = asr_chars / max(m["duration"], 0.1)
        rr = repeat_ratio(asr_text_pf)

        # ASR (SenseVoice) - 可选
        if asr_sv is not None and not skip_asr:
            asr_text_sv = _transcribe(asr_sv, samples)
        else:
            asr_text_sv = ""

        # 5 个新信号
        if skip_asr:
            silero_ratio = 0.0
            spec_ent = float("nan")
            f0_ratio = 0.0
            dual_cer = float("nan")
            rms_dyn = float("nan")
        else:
            try: silero_ratio = silero_speech_ratio(samples, sr)
            except Exception: silero_ratio = float("nan")
            try: spec_ent = spectral_entropy(samples)
            except Exception: spec_ent = float("nan")
            try: f0_ratio = f0_voiced_ratio(samples, sr)
            except Exception: f0_ratio = float("nan")
            try: rms_dyn = rms_dynamics(samples, sr)
            except Exception: rms_dyn = float("nan")
            if asr_sv is not None and asr_text_sv:
                dual_cer = pinyin_cer(asr_text_pf, asr_text_sv)
            else:
                dual_cer = float("nan")

        # 分桶
        tier, reason, n_pass, pass_list = classify_v2(
            m, asr_text_pf, asr_chars, asr_chars_per_sec,
            silero_ratio, spec_ent, f0_ratio, dual_cer, rms_dyn,
        )
        new_pass_dist[n_pass] += 1

        # 写 row
        row.update({
            "tier": tier, "reason": reason,
            "n_new_passed": n_pass, "new_passed_list": ",".join(pass_list),
            "asr_text_pf": asr_text_pf,
            "asr_chars": asr_chars,
            "asr_chars_per_sec": round(asr_chars_per_sec, 3),
            "repeat_ratio": round(rr, 3),
            "silero_speech_ratio": round(silero_ratio, 3) if not np.isnan(silero_ratio) else "",
            "spectral_entropy": round(spec_ent, 3) if not np.isnan(spec_ent) else "",
            "f0_voiced_ratio": round(f0_ratio, 3) if not np.isnan(f0_ratio) else "",
            "rms_dynamics": round(rms_dyn, 3) if not np.isnan(rms_dyn) else "",
            "dual_pinyin_cer": round(dual_cer, 3) if not np.isnan(dual_cer) else "",
            "asr_text_sv": asr_text_sv,
        })
        writer.writerow(row)
        tier_counts[tier] += 1

        # 复制
        if not args.no_copy:
            try:
                shutil.copy2(path, os.path.join(args.output, tier, fname))
            except Exception:
                pass

        # 进度
        if idx % 50 == 0:
            csv_file.flush()
            elapsed = time.time() - t_start
            rate = idx / elapsed
            remaining = (total - idx) / rate / 60
            tier_str = " ".join(f"{t}={tier_counts[t]}" for t in tiers if tier_counts[t])
            print(f"  [{idx:>4}/{total}] {rate:.1f} 条/秒 剩余 {remaining:.1f}分 | {tier_str}")

    csv_file.close()
    elapsed = (time.time() - t_start) / 60

    # 汇总
    print(f"\n{'='*78}")
    print(f"✓ 完成，总耗时 {elapsed:.1f} 分钟\n")
    print(f"📊 分桶 (共 {total}):")
    for t in tiers:
        c = tier_counts[t]
        print(f"  {t:<10} {c:>5}  ({c/total*100:5.1f}%)")
    print(f"\n📊 新信号通过数分布 (仅适用于过了原 green 的样本):")
    for n in range(6):
        c = new_pass_dist[n]
        if c > 0:
            print(f"  {n}/5 通过:  {c:>4}")

    # summary
    with open(os.path.join(args.output, "summary_v2.txt"), "w", encoding="utf-8") as f:
        f.write(f"clean_dataset_v2 - 增强版数据集清洗\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"输入: {args.input}\n")
        f.write(f"总数: {total}\n")
        f.write(f"耗时: {elapsed:.1f} 分钟\n\n")
        f.write("分桶:\n")
        for t in tiers:
            c = tier_counts[t]
            f.write(f"  {t}: {c} ({c/total*100:.1f}%)\n")
        f.write(f"\n新信号通过数分布:\n")
        for n in range(6):
            c = new_pass_dist[n]
            if c > 0:
                f.write(f"  {n}/5: {c}\n")
        f.write(f"\n阈值:\n")
        f.write(f"  silero >= {SILERO_SPEECH_RATIO_OK}\n")
        f.write(f"  spec_entropy <= {SPECTRAL_ENTROPY_MAX}\n")
        f.write(f"  f0_voiced >= {F0_VOICED_RATIO_OK}\n")
        f.write(f"  dual_pinyin_cer <= {DUAL_ASR_CER_MAX}\n")
        f.write(f"  rms_dynamics >= {RMS_DYNAMICS_MIN}\n")

    print(f"\n📁 报告: {csv_path}")
    print(f"📁 各 tier 子目录在: {args.output}/")
    print(f"\n👂 推荐做法:")
    print(f"   1. 抽 30 条 green++ 听一下，看是否还有听不懂的")
    print(f"   2. 如果还有 → 调严阈值（脚本顶部 SILERO_/F0_/DUAL_ASR_/SPECTRAL_/RMS_）重跑")
    print(f"   3. 后续算法都基于 green++ 评估")


if __name__ == "__main__":
    main()
