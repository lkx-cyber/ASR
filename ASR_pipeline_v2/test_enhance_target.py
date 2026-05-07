"""
目标说话人增强测试

抽样 N 条录音，跑 enhance_target_speaker，保存原始/增强对照 wav，
同时跑 FunASR 比较转写差异。

用法:
    # 默认从 verify_multi/ (上一步判定为多人的) 抽 20 条
    python test_enhance_target.py

    # 从其他目录抽
    python test_enhance_target.py --source recordings_cleaned/green --n 30

    # 不跑 ASR (只生成 wav)
    python test_enhance_target.py --no-asr

输出:
    test_enhance_target/
        {filename}/
            1_原始.wav
            2_增强后.wav
        report.csv               每条的 ASR 对比 + 增强参数
        summary.txt              总体统计
"""
import argparse
import csv
import glob
import os
import random
import re
import shutil
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio, save_audio
from enhance_target import enhance_target_speaker
from device_utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SR = 16000


def normalize_text(s):
    return re.sub(r"[^\w一-鿿]", "", s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=os.path.join(BASE_DIR, "verify_multi"),
                        help="抽样源目录")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "test_enhance_target"))
    parser.add_argument("--no-asr", action="store_true",
                        help="跳过 ASR 比较（仅生成 wav）")
    args = parser.parse_args()

    # 收集
    if not os.path.isdir(args.source):
        print(f"源目录不存在: {args.source}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.source, "*.wav")))
    if not files:
        print(f"没找到 wav: {args.source}")
        sys.exit(1)

    random.seed(args.seed)
    sampled = random.sample(files, min(args.n, len(files)))
    print(f"抽样 {len(sampled)} / {len(files)} 条 (seed={args.seed})\n")

    os.makedirs(args.out, exist_ok=True)

    # ASR
    asr = None
    if not args.no_asr:
        device = get_device()
        print(f"加载 FunASR ({device})...")
        from funasr import AutoModel
        asr = AutoModel(
            model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
            disable_update=True, disable_log=True, disable_pbar=True,
            device=device,
        )

    def transcribe(samples):
        if asr is None:
            return "", 0
        try:
            r = asr.generate(input=samples, fs=SR, batch_size_s=300)
            text = ""
            for item in r:
                if item.get("sentence_info"):
                    for sent in item["sentence_info"]:
                        text += sent.get("text", "")
                else:
                    text += item.get("text", "")
            text = text.strip()
            return text, len(normalize_text(text))
        except Exception:
            return "", 0

    # CSV
    csv_path = os.path.join(args.out, "report.csv")
    fields = [
        "filename", "duration", "action", "reason",
        "min_sim", "mean_sim", "n_windows", "n_silent",
        "raw_asr", "raw_chars", "enh_asr", "enh_chars",
        "char_diff", "rms_raw", "rms_enh", "rms_ratio",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    print("开始处理...\n")
    t_start = time.time()

    actions = {"enhance": 0, "passthrough": 0, "skip": 0}

    for idx, path in enumerate(sampled, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        item_dir = os.path.join(args.out, name)
        os.makedirs(item_dir, exist_ok=True)

        # 加载
        try:
            raw, sr = load_audio(path, verbose=False)
        except Exception as e:
            print(f"[{idx}/{len(sampled)}] {name}: 加载失败 {e}")
            continue

        dur = len(raw) / sr

        # 增强
        enhanced, info = enhance_target_speaker(raw, sr)
        actions[info["action"]] = actions.get(info["action"], 0) + 1

        # 保存 wav
        save_audio(os.path.join(item_dir, "1_原始.wav"), raw, sr)
        save_audio(os.path.join(item_dir, "2_增强后.wav"), enhanced, sr)

        # ASR
        raw_text, raw_chars = transcribe(raw)
        enh_text, enh_chars = transcribe(enhanced)

        rms_raw = float(np.sqrt(np.mean(raw ** 2)))
        rms_enh = float(np.sqrt(np.mean(enhanced ** 2)))

        row = {
            "filename": name,
            "duration": round(dur, 2),
            "action": info.get("action", ""),
            "reason": info.get("reason", ""),
            "min_sim": round(info["min_sim"], 3) if "min_sim" in info and not np.isnan(info["min_sim"]) else "",
            "mean_sim": round(info["mean_sim"], 3) if "mean_sim" in info and not np.isnan(info["mean_sim"]) else "",
            "n_windows": info.get("n_windows", ""),
            "n_silent": info.get("n_silent", ""),
            "raw_asr": raw_text,
            "raw_chars": raw_chars,
            "enh_asr": enh_text,
            "enh_chars": enh_chars,
            "char_diff": enh_chars - raw_chars,
            "rms_raw": round(rms_raw, 4),
            "rms_enh": round(rms_enh, 4),
            "rms_ratio": round(rms_enh / max(rms_raw, 1e-9), 3),
        }
        writer.writerow(row)
        csv_file.flush()

        # 打印
        diff_marker = "↑" if row["char_diff"] > 0 else ("↓" if row["char_diff"] < 0 else "=")
        print(f"[{idx:>2}/{len(sampled)}] {name[:14]} ({dur:.1f}s) "
              f"{info['action']:<11} sim={row['min_sim']}/{row['mean_sim']}  "
              f"raw={raw_chars}字 enh={enh_chars}字 {diff_marker}")
        if raw_text != enh_text:
            print(f"        raw: {raw_text[:60]}")
            print(f"        enh: {enh_text[:60]}")

    csv_file.close()

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"✓ 完成，耗时 {elapsed:.1f} 分钟")
    print(f"\n📊 行为分布:")
    for act, cnt in actions.items():
        if cnt > 0:
            print(f"  {act:<12} {cnt} 条")

    # 增强后字数变化统计
    csv_file = open(csv_path, "r", encoding="utf-8-sig")
    reader = csv.DictReader(csv_file)
    rows = list(reader)
    csv_file.close()

    char_diffs = [int(r["char_diff"]) for r in rows if r["char_diff"]]
    if char_diffs:
        n_better = sum(1 for d in char_diffs if d > 0)
        n_worse = sum(1 for d in char_diffs if d < 0)
        n_same = sum(1 for d in char_diffs if d == 0)
        print(f"\n📊 ASR 字数变化（增强 vs 原始）:")
        print(f"  字数变多: {n_better}  ({n_better/len(char_diffs)*100:.0f}%)")
        print(f"  字数变少: {n_worse}   ({n_worse/len(char_diffs)*100:.0f}%)")
        print(f"  字数相同: {n_same}    ({n_same/len(char_diffs)*100:.0f}%)")
        print(f"  平均变化: {np.mean(char_diffs):+.1f} 字")

    print(f"\n📁 输出位置: {args.out}/")
    print(f"   - 每条样本一个子目录，含 1_原始.wav 和 2_增强后.wav")
    print(f"   - report.csv 完整指标")
    print(f"\n👂 下一步: 听各子目录的 wav 对比，重点听:")
    print(f"   1. 增强后背景人声是否变弱了？")
    print(f"   2. 主说话人内容是否完整？(没被吞字)")
    print(f"   3. 听感是否变差? (机械感、电音)")

    # summary
    with open(os.path.join(args.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"目标说话人增强测试 ({len(sampled)} 条)\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"源: {args.source}\n")
        f.write(f"耗时: {elapsed:.1f} 分钟\n\n")
        f.write("行为分布:\n")
        for act, cnt in actions.items():
            f.write(f"  {act}: {cnt}\n")
        if char_diffs:
            f.write(f"\nASR 字数变化:\n")
            f.write(f"  增多: {n_better}, 减少: {n_worse}, 不变: {n_same}\n")
            f.write(f"  平均: {np.mean(char_diffs):+.1f} 字\n")


if __name__ == "__main__":
    main()
