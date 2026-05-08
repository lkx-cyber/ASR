"""
测试 is_multispeaker 判别器的准确率

方法：
  1. 假设 100 条 ground truth 样本都是单人（来自 green++ 干净数据，业务 95% 是单人 PTT）
     → 跑判别器，看多少被误判为「多人」（False Positive 率）
  2. 跑 verify_multi/ 61 条（之前判定为多人的）作 sanity check
     注意：这是 circular 测试，不能反映真实准确率
  3. 列出可疑误判样本供人耳校验

输出：
  multispeaker_accuracy_<时间戳>/
    report.csv         每条样本的判别结果 + 假设标签
    suspect_fp.txt     疑似 False Positive 列表（GT 单人但判为多人）
    suspect_fn.txt     疑似 False Negative 列表（verify_multi 但判为单人）
    summary.txt        准确率估计 + 警告
"""
import argparse
import csv
import os
import shutil
import sys
import time
import warnings
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-csv", default="groundtruth.csv")
    parser.add_argument("--gt-audio", default="recordings_cleaned_v2/green++")
    parser.add_argument("--multi-dir", default="verify_multi")
    args = parser.parse_args()

    gt_csv = os.path.join(BASE_DIR, args.gt_csv)
    gt_audio_dir = os.path.join(BASE_DIR, args.gt_audio)
    multi_dir = os.path.join(BASE_DIR, args.multi_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(BASE_DIR, f"multispeaker_accuracy_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # 加载判别器
    print("加载 is_multispeaker（resemblyzer）...")
    from is_multispeaker import classify
    print("  ✓\n")

    # ---------- 1. 假设 GT 全是单人 ----------
    with open(gt_csv, encoding="utf-8-sig") as f:
        gt_rows = [r for r in csv.DictReader(f) if r["correct_text"].strip()
                   and r["correct_text"].strip() != "/n"]
    print(f"📚 GT 样本: {len(gt_rows)} 条 (假设全部为单人)")

    gt_results = []
    fp_count = 0
    for i, r in enumerate(gt_rows, 1):
        path = os.path.join(gt_audio_dir, r["filename"])
        if not os.path.exists(path):
            continue
        try:
            res = classify(path)
        except Exception as e:
            print(f"  ⚠️ {r['filename']}: {e}")
            continue
        is_multi = bool(res.get("is_multi"))
        if is_multi:
            fp_count += 1
        gt_results.append({
            "filename": r["filename"],
            "assumed_label": "single",
            "predicted_multi": is_multi,
            "min_sim": res.get("min_sim"),
            "mean_sim": res.get("mean_sim"),
            "n_windows": res.get("n_windows"),
            "speech_ratio": res.get("speech_ratio"),
            "reason": res.get("reason"),
            "correct_text": r["correct_text"][:50],
        })
        if i % 20 == 0:
            print(f"  GT 进度 {i}/{len(gt_rows)}  当前 FP={fp_count}")
    n_gt = len(gt_results)
    fp_rate = fp_count / max(1, n_gt)
    print(f"  GT: {n_gt} 条, 误判为多人 {fp_count} 条 (FP rate ≈ {fp_rate*100:.1f}%)\n")

    # ---------- 2. verify_multi (circular sanity check) ----------
    multi_results = []
    fn_count = 0
    if os.path.isdir(multi_dir):
        multi_files = sorted([f for f in os.listdir(multi_dir) if f.endswith(".wav")])
        print(f"📚 verify_multi: {len(multi_files)} 条 (假设全部为多人，⚠️ circular!)")
        for i, fname in enumerate(multi_files, 1):
            path = os.path.join(multi_dir, fname)
            try:
                res = classify(path)
            except Exception as e:
                print(f"  ⚠️ {fname}: {e}")
                continue
            is_multi = bool(res.get("is_multi"))
            if not is_multi:
                fn_count += 1
            multi_results.append({
                "filename": fname,
                "assumed_label": "multi",
                "predicted_multi": is_multi,
                "min_sim": res.get("min_sim"),
                "mean_sim": res.get("mean_sim"),
                "n_windows": res.get("n_windows"),
                "speech_ratio": res.get("speech_ratio"),
                "reason": res.get("reason"),
                "correct_text": "",
            })
            if i % 20 == 0:
                print(f"  Multi 进度 {i}/{len(multi_files)}  当前 FN={fn_count}")
        n_multi = len(multi_results)
        fn_rate = fn_count / max(1, n_multi)
        print(f"  Multi: {n_multi} 条, 判为单人 {fn_count} 条 (FN rate ≈ {fn_rate*100:.1f}%)\n")

    # ---------- 输出 ----------
    all_rows = gt_results + multi_results
    # report.csv
    report_path = os.path.join(out_dir, "report.csv")
    if all_rows:
        with open(report_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # suspect_fp.txt（GT 里被判为多人）
    fp_path = os.path.join(out_dir, "suspect_fp.txt")
    with open(fp_path, "w", encoding="utf-8") as f:
        f.write("# 假设单人 → 判别器说多人 (疑似 False Positive)\n")
        f.write("# 请人耳听一下确认，可能：\n")
        f.write("#   - 真的是多人（GT 假设错了）→ 不算误判\n")
        f.write("#   - 真的是单人 → 是误判，FP rate 准确\n")
        f.write("#   - 边缘 case → 看具体业务判断\n\n")
        f.write(f"{'文件':<40} {'min_sim':<10} {'reason':<40} {'文本'}\n")
        f.write("-" * 130 + "\n")
        for r in gt_results:
            if r["predicted_multi"]:
                f.write(f"{r['filename']:<40} "
                        f"{r['min_sim'] if r['min_sim'] is not None else '':<10}"
                        f"{r['reason'] or '':<40} {r['correct_text']}\n")

    # suspect_fn.txt
    fn_path = os.path.join(out_dir, "suspect_fn.txt")
    with open(fn_path, "w", encoding="utf-8") as f:
        f.write("# verify_multi 中被判别器说\"单人\"的（疑似 False Negative）\n")
        f.write("# 注：这是 circular 测试，仅供参考\n\n")
        f.write(f"{'文件':<40} {'min_sim':<10} {'reason'}\n")
        f.write("-" * 90 + "\n")
        for r in multi_results:
            if not r["predicted_multi"]:
                f.write(f"{r['filename']:<40} "
                        f"{r['min_sim'] if r['min_sim'] is not None else '':<10}"
                        f"{r['reason'] or ''}\n")

    # summary.txt
    n_multi = len(multi_results)
    n_total = n_gt + n_multi
    tp = n_multi - fn_count
    fp = fp_count
    tn = n_gt - fp_count
    fn_n = fn_count
    accuracy = (tp + tn) / max(1, n_total)

    summary_path = os.path.join(out_dir, "summary.txt")
    lines = []
    lines.append("=" * 72)
    lines.append(f"is_multispeaker 准确率测试  ({timestamp})")
    lines.append("=" * 72)
    lines.append("")
    lines.append("【数据集】")
    lines.append(f"  GT (假设单人):    {n_gt} 条 (来自 {args.gt_audio})")
    lines.append(f"  verify_multi:    {n_multi} 条 (来自 {args.multi_dir})")
    lines.append(f"  ⚠️  verify_multi 是判别器自己生成的，FN rate 数据 circular")
    lines.append("")
    lines.append("【混淆矩阵】（基于上述假设标签）")
    lines.append("")
    lines.append(f"               预测单人    预测多人")
    lines.append(f"   假设单人      TN={tn:<6}    FP={fp:<6}")
    lines.append(f"   假设多人      FN={fn_n:<6}    TP={tp:<6}")
    lines.append("")
    lines.append("【关键指标】")
    lines.append(f"  整体准确率:       {accuracy*100:.1f}%  (=({tp}+{tn})/{n_total})")
    lines.append(f"  单人误判率(FP):   {fp_rate*100:.1f}%  ← 关键!")
    lines.append(f"    (单人被判成多人 → 错误地走分离 → 引入分离副作用)")
    if n_multi:
        lines.append(f"  多人漏判率(FN):   {fn_rate*100:.1f}%  (circular, 仅参考)")
        lines.append(f"    (多人被判成单人 → 跳过分离 → 失去分离收益)")
    lines.append("")
    lines.append("【路由收益门槛分析】")
    lines.append(f"  业务真实分布: ~95% 单人, ~5% 多人")
    lines.append(f"  路由方案: 单人→M3b (raw+denoise), 多人→M5 (sep)")
    lines.append("")
    lines.append(f"  路由 vs 一刀切 M3b:")
    lines.append(f"    收益: 5% 多人正确路由 → 用 M5 SNR +5dB")
    lines.append(f"    损失: 95% × {fp_rate*100:.1f}% = {95*fp_rate:.1f}% 单人误路由 → 用 M5 引入分离副作用")
    lines.append("")
    if fp_rate > 0.05:
        lines.append(f"  ⚠️  FP rate {fp_rate*100:.1f}% > 5% 警戒线")
        lines.append(f"     单人误判损失大于多人路由收益的可能性高")
        lines.append(f"     → 不建议立即上线路由，先调判别器阈值或扩大判别能力")
    else:
        lines.append(f"  ✓  FP rate {fp_rate*100:.1f}% ≤ 5%, 路由有可能正收益")
        lines.append(f"     → 可以做 30 条 A/B 实测验证")
    lines.append("")
    lines.append("【输出文件】")
    lines.append(f"  report.csv        所有样本的判别明细")
    lines.append(f"  suspect_fp.txt    疑似 FP（请人耳听）")
    lines.append(f"  suspect_fn.txt    疑似 FN（仅参考）")

    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print()
    print(f"✓ 输出: {out_dir}")


if __name__ == "__main__":
    main()
