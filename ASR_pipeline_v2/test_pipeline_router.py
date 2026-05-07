"""
测试路由 pipeline：抽样验证「raw → FunASR」vs「路由方案」哪种文本更准。

输出:
    test_pipeline_router/
        {filename}/
            raw.wav                  原始
            track_0.wav, track_1.wav (多人时) baseline 分离结果
            selected_track.wav        (多人时) 选中的目标轨
        report.csv                   每条的对比
        summary.txt                  分支统计 + 字数对比
"""
import argparse
import csv
import glob
import os
import random
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio, save_audio
from pipeline_router import RouterPipeline
from separate_baseline import SR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="recordings_cleaned/green",
                        help="音频源目录")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="test_pipeline_router")
    parser.add_argument("--save-wavs", action="store_true",
                        help="保存原始 + 分离轨 wav (用于人耳校验)")
    args = parser.parse_args()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    src_dir = args.source if os.path.isabs(args.source) else os.path.join(BASE_DIR, args.source)
    out_dir = args.out if os.path.isabs(args.out) else os.path.join(BASE_DIR, args.out)

    if not os.path.isdir(src_dir):
        print(f"源目录不存在: {src_dir}")
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(src_dir, "*.wav")))
    if not files:
        print(f"没找到 wav: {src_dir}")
        sys.exit(1)

    random.seed(args.seed)
    sampled = random.sample(files, min(args.n, len(files)))
    print(f"抽样 {len(sampled)}/{len(files)} 条 (seed={args.seed})\n")

    os.makedirs(out_dir, exist_ok=True)

    # 加载 pipeline
    pipeline = RouterPipeline()

    # CSV
    csv_path = os.path.join(out_dir, "report.csv")
    fields = [
        "filename", "duration", "branch",
        "is_multi", "min_sim", "n_windows",
        "selected_track", "track_scores",
        "raw_text", "raw_chars",
        "router_text", "router_chars",
        "char_diff", "text_match",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    branch_counts = {"single": 0, "multi": 0, "multi_separation_failed": 0}
    char_diffs = []
    text_matches = 0

    print("开始处理...\n")
    t_start = time.time()

    for idx, path in enumerate(sampled, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            raw, sr = load_audio(path, verbose=False)
        except Exception as e:
            print(f"[{idx}/{len(sampled)}] {name} 加载失败: {e}")
            continue

        dur = len(raw) / sr

        # 路由结果
        result = pipeline.process(raw, sr)

        # raw 直转结果（对照）
        raw_text = pipeline.transcribe(raw)

        router_text = result["text"]
        import re
        norm = lambda s: re.sub(r"[^\w一-鿿]", "", s)
        raw_chars = len(norm(raw_text))
        router_chars = len(norm(router_text))

        # 保存 wav
        if args.save_wavs:
            d = os.path.join(out_dir, name)
            os.makedirs(d, exist_ok=True)
            save_audio(os.path.join(d, "raw.wav"), raw, sr)
            if result.get("tracks") is not None:
                for i, t in enumerate(result["tracks"]):
                    save_audio(os.path.join(d, f"track_{i}.wav"), t, sr)
                save_audio(
                    os.path.join(d, f"selected_track_{result['selected_track']}.wav"),
                    result["tracks"][result["selected_track"]], sr
                )

        # 统计
        branch = result["branch"]
        branch_counts[branch] = branch_counts.get(branch, 0) + 1
        char_diff = router_chars - raw_chars
        char_diffs.append(char_diff)
        text_match = (raw_text.strip() == router_text.strip())
        if text_match:
            text_matches += 1

        is_multi_info = result["is_multi_info"]
        row = {
            "filename": name,
            "duration": round(dur, 2),
            "branch": branch,
            "is_multi": is_multi_info.get("is_multi"),
            "min_sim": round(is_multi_info["min_sim"], 3) if not np.isnan(is_multi_info.get("min_sim", float("nan"))) else "",
            "n_windows": is_multi_info.get("n_windows"),
            "selected_track": result.get("selected_track"),
            "track_scores": str(result.get("track_scores")) if result.get("track_scores") else "",
            "raw_text": raw_text,
            "raw_chars": raw_chars,
            "router_text": router_text,
            "router_chars": router_chars,
            "char_diff": char_diff,
            "text_match": text_match,
        }
        writer.writerow(row)
        csv_file.flush()

        diff_marker = "↑" if char_diff > 0 else ("↓" if char_diff < 0 else "=")
        print(f"[{idx:>2}/{len(sampled)}] {name[:14]} ({dur:.1f}s) "
              f"{branch:<10}  raw={raw_chars}字 router={router_chars}字 {diff_marker}")
        if not text_match:
            print(f"      raw:    {raw_text[:65]}")
            print(f"      router: {router_text[:65]}")

    csv_file.close()

    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*70}")
    print(f"✓ 完成，耗时 {elapsed:.1f} 分钟\n")

    print("📊 路由分支分布:")
    for br, cnt in branch_counts.items():
        if cnt > 0:
            print(f"  {br:<28} {cnt} ({cnt/len(sampled)*100:.0f}%)")

    print(f"\n📊 router vs raw 文本:")
    print(f"  完全相同: {text_matches}/{len(sampled)}  ({text_matches/len(sampled)*100:.0f}%)")
    if char_diffs:
        n_more = sum(1 for d in char_diffs if d > 0)
        n_less = sum(1 for d in char_diffs if d < 0)
        print(f"  router 字数变多: {n_more}")
        print(f"  router 字数变少: {n_less}")
        print(f"  平均字数变化: {np.mean(char_diffs):+.1f}")

    print(f"\n📁 详细报告: {csv_path}")
    if args.save_wavs:
        print(f"📁 各样本子目录: {out_dir}/")
    print("\n👂 听感校验建议（看 CSV，重点听 text_match=False 的）:")
    print("    1. 多人分支：listening selected_track，是否真的是按按钮的孩子")
    print("    2. 跟 raw 文本比较，哪个更接近你听到的目标人内容")

    # summary
    with open(os.path.join(out_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"路由 pipeline 测试 ({len(sampled)} 条)\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"源: {src_dir}\n")
        f.write(f"耗时: {elapsed:.1f} 分钟\n\n")
        f.write("分支分布:\n")
        for br, cnt in branch_counts.items():
            f.write(f"  {br}: {cnt}\n")
        f.write(f"\n文本对比:\n")
        f.write(f"  完全相同: {text_matches}/{len(sampled)}\n")
        if char_diffs:
            f.write(f"  router 字数变多: {n_more}\n")
            f.write(f"  router 字数变少: {n_less}\n")
            f.write(f"  平均字数变化: {np.mean(char_diffs):+.1f}\n")


if __name__ == "__main__":
    main()
