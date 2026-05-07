"""
单/多说话人快速判别器

核心思路：
  1. 把音频按 1.5s 滑窗切片（0.5s 步长，跳过静音段）
  2. 每个窗算 Resemblyzer speaker embedding (256-dim)
  3. 计算窗与窗之间的余弦相似度
  4. 最低相似度 (min_sim) 是核心信号:
       min_sim > 0.80 → 单人 (所有窗都像同一个人)
       min_sim < 0.65 → 多人 (至少一对窗差异显著)
       0.65 - 0.80    → 边缘 (borderline)

输出 dict:
  {
    "is_multi": bool,
    "confidence": float (0-1),
    "min_sim": float,         # 最关键
    "mean_sim": float,
    "n_windows": int,
    "duration": float,
    "speech_ratio": float,
    "reason": str,
  }

用法:
    # 单文件
    python is_multispeaker.py recordings_cleaned/green/abc.wav

    # 批量评估一个目录
    python is_multispeaker.py --dir recordings_cleaned/green
    python is_multispeaker.py --dir recordings_cleaned/green --save-multi-to verify_multi/

    # 作为模块
    from is_multispeaker import classify
    r = classify("audio.wav")
    print(r["is_multi"], r["min_sim"])
"""
import argparse
import csv
import glob
import json
import os
import shutil
import sys
import time
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio
from device_utils import get_voice_encoder

SR = 16000

# 阈值（可调）
MIN_DURATION_FOR_RELIABLE = 1.5  # 短于此秒数时无法可靠判别
MIN_SPEECH_RATIO = 0.20           # 静音过多时不可靠
WIN_SEC = 1.5                     # 滑窗大小
HOP_SEC = 0.5                     # 步长
WIN_MIN_RMS = 0.005               # 太安静的窗跳过

THRESHOLD_MULTI = 0.65            # min_sim < 此值 → 多人
THRESHOLD_SINGLE = 0.80           # min_sim > 此值 → 单人


# ---------- VAD ----------
def vad_energy(samples, frame_ms=20, threshold_db=-40):
    """简单能量 VAD"""
    frame_size = int(SR * frame_ms / 1000)
    n = len(samples) // frame_size
    if n == 0:
        return np.array([], dtype=bool)
    frames = samples[:n * frame_size].reshape(n, frame_size)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-10)
    return db > threshold_db


# ---------- 滑窗 embedding ----------
def compute_window_embeddings(samples, win_sec=WIN_SEC, hop_sec=HOP_SEC):
    """
    滑窗算 speaker embeddings。返回 (embs, win_starts)
    跳过 RMS 太低的窗。
    """
    from resemblyzer import preprocess_wav
    encoder = get_voice_encoder()

    win = int(win_sec * SR)
    hop = int(hop_sec * SR)
    embs = []
    starts = []

    for start in range(0, max(1, len(samples) - win + 1), hop):
        chunk = samples[start:start + win]
        if len(chunk) < SR * 0.6:
            continue
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < WIN_MIN_RMS:
            continue
        try:
            wav = preprocess_wav(chunk, source_sr=SR)
            if len(wav) < SR * 0.4:
                continue
            emb = encoder.embed_utterance(wav)
            embs.append(emb)
            starts.append(start / SR)
        except Exception:
            continue

    if not embs:
        return np.zeros((0, 256), dtype=np.float32), []
    return np.stack(embs).astype(np.float32), starts


# ---------- 主判别函数 ----------
def classify(samples_or_path, sr=SR, return_embs=False):
    """
    单/多人判别。
    Args:
      samples_or_path: numpy array (mono float32 16k) 或 文件路径
      sr: 当传 numpy 时的采样率
      return_embs: 是否返回 embedding 矩阵和时间戳（调试用）
    """
    if isinstance(samples_or_path, str):
        samples, sr = load_audio(samples_or_path, verbose=False)
    else:
        samples = np.asarray(samples_or_path, dtype=np.float32)

    duration = len(samples) / sr

    # 太短：直接判单人
    if duration < MIN_DURATION_FOR_RELIABLE:
        return _make_result(
            is_multi=False, confidence=0.5,
            min_sim=float("nan"), mean_sim=float("nan"),
            n_windows=0, duration=duration, speech_ratio=0.0,
            reason=f"音频过短 ({duration:.2f}s < {MIN_DURATION_FOR_RELIABLE}s)，默认单人",
        )

    # VAD 检查
    is_speech = vad_energy(samples)
    speech_ratio = float(is_speech.mean()) if len(is_speech) > 0 else 0.0
    if speech_ratio < MIN_SPEECH_RATIO:
        return _make_result(
            is_multi=False, confidence=0.7,
            min_sim=float("nan"), mean_sim=float("nan"),
            n_windows=0, duration=duration, speech_ratio=speech_ratio,
            reason=f"语音占比过低 ({speech_ratio*100:.0f}% < {MIN_SPEECH_RATIO*100:.0f}%)，默认单人",
        )

    # 滑窗算 embeddings
    embs, starts = compute_window_embeddings(samples)
    if len(embs) < 2:
        return _make_result(
            is_multi=False, confidence=0.6,
            min_sim=float("nan"), mean_sim=float("nan"),
            n_windows=len(embs), duration=duration, speech_ratio=speech_ratio,
            reason=f"有效窗口不足 ({len(embs)} 个 < 2)，默认单人",
            embs=embs, starts=starts, return_embs=return_embs,
        )

    # 余弦相似度矩阵
    norms = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12)
    sims = norms @ norms.T
    upper = sims[np.triu_indices(len(sims), k=1)]
    min_sim = float(upper.min())
    mean_sim = float(upper.mean())

    # 决策
    if min_sim < THRESHOLD_MULTI:
        is_multi = True
        confidence = float(min(1.0, (THRESHOLD_SINGLE - min_sim) / (THRESHOLD_SINGLE - THRESHOLD_MULTI)))
        reason = f"min_sim={min_sim:.3f} < {THRESHOLD_MULTI}，存在显著不同的窗"
    elif min_sim > THRESHOLD_SINGLE:
        is_multi = False
        confidence = float(min(1.0, (min_sim - THRESHOLD_MULTI) / (THRESHOLD_SINGLE - THRESHOLD_MULTI)))
        reason = f"min_sim={min_sim:.3f} > {THRESHOLD_SINGLE}，所有窗高度相似"
    else:
        # borderline：偏向 < 0.725 判多人
        is_multi = min_sim < (THRESHOLD_MULTI + THRESHOLD_SINGLE) / 2
        confidence = 0.5
        reason = f"min_sim={min_sim:.3f} 在 [{THRESHOLD_MULTI},{THRESHOLD_SINGLE}] 边缘区"

    return _make_result(
        is_multi=is_multi, confidence=confidence,
        min_sim=min_sim, mean_sim=mean_sim,
        n_windows=len(embs), duration=duration, speech_ratio=speech_ratio,
        reason=reason,
        embs=embs, starts=starts, return_embs=return_embs,
    )


def _make_result(**kwargs):
    return_embs = kwargs.pop("return_embs", False)
    embs = kwargs.pop("embs", None)
    starts = kwargs.pop("starts", None)
    result = kwargs
    if return_embs:
        result["embs"] = embs
        result["starts"] = starts
    return result


# ---------- 批处理 ----------
def batch_classify(files, save_multi_to=None, save_single_to=None, verbose=True):
    results = []
    t0 = time.time()
    for i, path in enumerate(files, 1):
        try:
            r = classify(path)
            r["file"] = os.path.basename(path)
        except Exception as e:
            r = {"file": os.path.basename(path), "error": str(e),
                 "is_multi": False, "confidence": 0.0,
                 "min_sim": float("nan"), "mean_sim": float("nan"),
                 "n_windows": 0, "duration": 0.0, "speech_ratio": 0.0,
                 "reason": f"error: {e}"}
        results.append(r)

        # 可选：复制多人/单人样本到验证目录
        if save_multi_to and r.get("is_multi"):
            os.makedirs(save_multi_to, exist_ok=True)
            shutil.copy(path, os.path.join(save_multi_to, os.path.basename(path)))
        if save_single_to and not r.get("is_multi") and not r.get("error"):
            os.makedirs(save_single_to, exist_ok=True)
            shutil.copy(path, os.path.join(save_single_to, os.path.basename(path)))

        if verbose and i % 50 == 0:
            elapsed = time.time() - t0
            rate = i / elapsed
            print(f"  [{i}/{len(files)}] {rate:.1f} 条/秒, "
                  f"剩余 {(len(files)-i)/rate/60:.1f} 分钟")

    return results


def write_csv(results, csv_path):
    fields = ["file", "is_multi", "confidence", "min_sim", "mean_sim",
              "n_windows", "duration", "speech_ratio", "reason"]
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fields}
            for k, v in row.items():
                if isinstance(v, float) and np.isnan(v):
                    row[k] = ""
            w.writerow(row)


def print_summary(results):
    n_total = len(results)
    n_err = sum(1 for r in results if r.get("error"))
    n_multi = sum(1 for r in results if r.get("is_multi") and not r.get("error"))
    n_single = n_total - n_multi - n_err

    print(f"\n{'='*60}")
    print(f"  总数: {n_total}")
    print(f"  🔵 单人: {n_single}  ({n_single/n_total*100:.1f}%)")
    print(f"  🔴 多人: {n_multi}   ({n_multi/n_total*100:.1f}%)")
    print(f"  ⚠️  错误: {n_err}    ({n_err/n_total*100:.1f}%)")

    # min_sim 分布
    sims = [r["min_sim"] for r in results
            if not r.get("error") and not np.isnan(r.get("min_sim", float("nan")))]
    if sims:
        sims = np.array(sims)
        print(f"\n  min_sim 分布:")
        print(f"    均值={sims.mean():.3f}  中位数={np.median(sims):.3f}")
        print(f"    P10={np.percentile(sims,10):.3f}  P90={np.percentile(sims,90):.3f}")

    # 高置信度多人 top-N (供你听感校验)
    multi_results = [r for r in results
                     if r.get("is_multi") and not r.get("error")]
    multi_results.sort(key=lambda r: r["confidence"], reverse=True)
    if multi_results:
        print(f"\n  Top 10 高置信度多人样本（建议人耳验证）:")
        for r in multi_results[:10]:
            print(f"    {r['file']}  conf={r['confidence']:.2f}  min_sim={r['min_sim']:.3f}")


# ---------- CLI ----------
def main():
    parser = argparse.ArgumentParser(description="单/多说话人判别")
    parser.add_argument("input", nargs="?", help="单个 wav 文件路径")
    parser.add_argument("--dir", help="批量处理目录（默认处理 *.wav）")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文件")
    parser.add_argument("--out-csv", default="multispeaker_report.csv",
                        help="批量模式输出 CSV 路径")
    parser.add_argument("--save-multi-to", default=None,
                        help="把判为多人的样本复制到该目录（供听感校验）")
    parser.add_argument("--save-single-to", default=None,
                        help="把判为单人的样本复制到该目录")
    args = parser.parse_args()

    # 单文件模式
    if args.input and not args.dir:
        if not os.path.exists(args.input):
            print(f"文件不存在: {args.input}")
            sys.exit(1)
        r = classify(args.input)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    # 批量模式
    if not args.dir:
        parser.print_help()
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.dir, "*.wav")))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"目录里没有 wav 文件: {args.dir}")
        sys.exit(1)

    print(f"处理 {len(files)} 个文件...\n")
    results = batch_classify(
        files,
        save_multi_to=args.save_multi_to,
        save_single_to=args.save_single_to,
    )
    write_csv(results, args.out_csv)
    print_summary(results)
    print(f"\n📁 详细报告: {args.out_csv}")
    if args.save_multi_to:
        n = sum(1 for r in results if r.get("is_multi"))
        print(f"📁 多人样本副本: {args.save_multi_to}/  ({n} 个)")


if __name__ == "__main__":
    main()
