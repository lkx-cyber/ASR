"""
分离效果测试脚本 (全量版)

对 recordings_cleaned/green + yellow 全部 ~3031 条录音批量跑分离评估。

为了控制时间，默认行为:
  - 只用 baseline (ONNX + 增强后处理)，跳过 MossFormer2
  - 只算自动指标 (相似度、能量比)，跳过 ASR (太慢)
  - 不保存输出 wav (会占 ~3GB)，只保存指标到 CSV

可选:
  --with-asr       加上 FunASR 转写两路输出（耗时 ~3 倍）
  --with-mossformer 也跑 MossFormer2（耗时 ~10 倍）
  --save-all       保存所有输出 wav（占用大量磁盘）
  --save-failures  仅保存疑似失败/边缘 case 的 wav (推荐)
  --tiers          指定桶 (默认 green yellow)

输出:
  test_separation_all/
    all_report.csv             全部录音的指标 + 自动分类
    all_summary.txt            分类统计 + 类型占比
    failures/                  (--save-failures) 疑似失败 case 的输出
      {filename}/
        mix.wav, 轨1.wav, 轨2.wav

预估耗时:
  baseline only, no asr:   ~30 min
  +asr:                    ~2 hours
  +mossformer +asr:        ~10 hours

建议第一次跑用默认参数（30 分钟），看分类分布后再决定要不要加 asr/mossformer。
"""
import os
import sys
import csv
import glob
import time
import argparse
import shutil
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio, save_audio
from separate_baseline import (
    rms_normalize, preemphasis, deemphasis,
    spectral_denoise, wiener_refine,
    evaluate_separation, speaker_similarity,
    MODEL_PATH, SR,
)
from test_separation_sample30 import (
    separate_baseline, separate_mossformer, classify_separation,
)
from device_utils import get_device, device_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
OUT_DIR = os.path.join(BASE_DIR, "test_separation_all")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-asr", action="store_true",
                        help="加上 FunASR 转写两路输出")
    parser.add_argument("--with-mossformer", action="store_true",
                        help="同时跑 MossFormer2 (10 倍耗时)")
    parser.add_argument("--save-all", action="store_true",
                        help="保存所有分离 wav (占大量磁盘)")
    parser.add_argument("--save-failures", action="store_true",
                        help="仅保存自动分类为 failed/ambiguous 的输出")
    parser.add_argument("--tiers", nargs="+", default=["green", "yellow"],
                        help="处理哪些桶")
    parser.add_argument("--limit", type=int, default=None,
                        help="只处理前 N 条 (调试用)")
    args = parser.parse_args()

    # 收集
    files = []
    for tier in args.tiers:
        for f in sorted(glob.glob(os.path.join(CLEANED_DIR, tier, "*.wav"))):
            files.append((tier, f))
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"未找到文件: {CLEANED_DIR}/[{','.join(args.tiers)}]/")
        return

    total = len(files)
    print(f"将处理 {total} 条 (来自 {args.tiers})")
    print(f"--with-asr={args.with_asr}, --with-mossformer={args.with_mossformer}, "
          f"--save-all={args.save_all}, --save-failures={args.save_failures}")

    os.makedirs(OUT_DIR, exist_ok=True)
    if args.save_failures or args.save_all:
        os.makedirs(os.path.join(OUT_DIR, "failures"), exist_ok=True)

    device = get_device()
    print(f"\n🖥️  Device: {device_info()}")

    # 加载模型
    print("加载 baseline ONNX (CPU, 按方案 A)...")
    session = ort.InferenceSession(MODEL_PATH)  # baseline 仍用 CPU

    cv = None
    tmp_dir = os.path.join(OUT_DIR, "_tmp")
    if args.with_mossformer:
        print(f"加载 MossFormer2_SS_16K ({device})...")
        from clearvoice import ClearVoice
        cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])

    asr = None
    if args.with_asr:
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
            result = asr.generate(input=samples, fs=SR, batch_size_s=300)
            text = ""
            for item in result:
                if item.get("sentence_info"):
                    for sent in item["sentence_info"]:
                        text += sent.get("text", "")
                else:
                    text += item.get("text", "")
            import re
            return text.strip(), len(re.sub(r"[^\w一-鿿]", "", text))
        except Exception:
            return "", 0

    # CSV
    csv_path = os.path.join(OUT_DIR, "all_report.csv")
    fields = [
        "filename", "tier", "duration",
        "baseline_sim", "baseline_e1", "baseline_e2",
        "baseline_n1", "baseline_n2", "baseline_class",
        "baseline_asr1", "baseline_asr2",
    ]
    if args.with_mossformer:
        fields += [
            "mf2_sim", "mf2_e1", "mf2_e2",
            "mf2_n1", "mf2_n2", "mf2_class",
            "mf2_asr1", "mf2_asr2",
        ]

    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    summary = {"baseline": {}, "mf2": {}}
    print("\n开始批处理...\n")
    t_start = time.time()

    for idx, (tier, path) in enumerate(files, 1):
        fname = os.path.splitext(os.path.basename(path))[0]

        try:
            mix, sr = load_audio(path, verbose=False)
        except Exception as e:
            row = {"filename": fname, "tier": tier, "baseline_class": f"load_error: {e}"}
            writer.writerow(_filter_row(row, fields))
            continue

        row = {"filename": fname, "tier": tier, "duration": round(len(mix) / SR, 2)}

        # === baseline ===
        try:
            b_sources = separate_baseline(session, mix)
            b_metrics = evaluate_separation(mix, b_sources)
            b_sim = speaker_similarity(b_sources)
            b_t1, b_n1 = transcribe(b_sources[0]) if asr else ("", 0)
            b_t2, b_n2 = transcribe(b_sources[1]) if asr else ("", 0)
            b_class, _ = classify_separation(
                b_sim if isinstance(b_sim, float) else 1.0,
                b_metrics["能量比"], b_n1, b_n2
            )
            row.update({
                "baseline_sim": round(b_sim, 3) if isinstance(b_sim, float) else "n/a",
                "baseline_e1": b_metrics["能量比"][0],
                "baseline_e2": b_metrics["能量比"][1],
                "baseline_n1": b_n1, "baseline_n2": b_n2,
                "baseline_class": b_class,
                "baseline_asr1": b_t1, "baseline_asr2": b_t2,
            })
            summary["baseline"][b_class] = summary["baseline"].get(b_class, 0) + 1

            # 保存 wav
            should_save = (
                args.save_all
                or (args.save_failures and b_class in ("failed_or_single", "ambiguous", "single_no_content"))
            )
            if should_save:
                d = os.path.join(OUT_DIR, "failures", fname)
                os.makedirs(d, exist_ok=True)
                save_audio(os.path.join(d, "mix.wav"), mix, SR)
                save_audio(os.path.join(d, "baseline_轨1.wav"), b_sources[0], SR)
                save_audio(os.path.join(d, "baseline_轨2.wav"), b_sources[1], SR)
        except Exception as e:
            row["baseline_class"] = f"error: {e}"

        # === MossFormer2 (可选) ===
        if cv is not None:
            try:
                m_sources = separate_mossformer(cv, mix, tmp_dir)
                m_metrics = evaluate_separation(mix, m_sources)
                m_sim = speaker_similarity(m_sources)
                m_t1, m_n1 = transcribe(m_sources[0]) if asr else ("", 0)
                m_t2, m_n2 = transcribe(m_sources[1]) if asr else ("", 0)
                m_class, _ = classify_separation(
                    m_sim if isinstance(m_sim, float) else 1.0,
                    m_metrics["能量比"], m_n1, m_n2
                )
                row.update({
                    "mf2_sim": round(m_sim, 3) if isinstance(m_sim, float) else "n/a",
                    "mf2_e1": m_metrics["能量比"][0],
                    "mf2_e2": m_metrics["能量比"][1],
                    "mf2_n1": m_n1, "mf2_n2": m_n2,
                    "mf2_class": m_class,
                    "mf2_asr1": m_t1, "mf2_asr2": m_t2,
                })
                summary["mf2"][m_class] = summary["mf2"].get(m_class, 0) + 1
            except Exception as e:
                row["mf2_class"] = f"error: {e}"

        writer.writerow(_filter_row(row, fields))

        if idx % 50 == 0:
            csv_file.flush()
            elapsed = time.time() - t_start
            rate = idx / elapsed
            remaining = (total - idx) / rate / 60
            classes_str = " ".join(
                f"{k}={v}" for k, v in sorted(summary["baseline"].items(), key=lambda x: -x[1])[:4]
            )
            print(f"  [{idx:>4}/{total}] {rate:.1f} 条/秒 剩余 {remaining:.1f} 分钟 | {classes_str}")

    csv_file.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 汇总
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*78}")
    print(f"✓ 完成，耗时 {elapsed:.1f} 分钟")
    print(f"\n📊 baseline 自动分类:")
    for cls, count in sorted(summary["baseline"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<22} {count:>5}  ({count/total*100:5.1f}%)")
    if cv is not None:
        print(f"\n📊 MossFormer2 自动分类:")
        for cls, count in sorted(summary["mf2"].items(), key=lambda x: -x[1]):
            print(f"  {cls:<22} {count:>5}  ({count/total*100:5.1f}%)")

    print(f"\n📁 详细数据: {csv_path}")
    if args.save_failures:
        n_saved = len(os.listdir(os.path.join(OUT_DIR, "failures")))
        print(f"📁 疑似失败 case 输出: {OUT_DIR}/failures/  ({n_saved} 个)")

    # summary.txt
    with open(os.path.join(OUT_DIR, "all_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"分离效果全量测试 - {total} 条\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"桶: {args.tiers}\n")
        f.write(f"配置: with_asr={args.with_asr}, with_mossformer={args.with_mossformer}\n")
        f.write(f"耗时: {elapsed:.1f} 分钟\n\n")
        f.write("Baseline 自动分类:\n")
        for cls, count in sorted(summary["baseline"].items(), key=lambda x: -x[1]):
            f.write(f"  {cls}: {count} ({count/total*100:.1f}%)\n")
        if cv is not None:
            f.write("\nMossFormer2 自动分类:\n")
            for cls, count in sorted(summary["mf2"].items(), key=lambda x: -x[1]):
                f.write(f"  {cls}: {count} ({count/total*100:.1f}%)\n")
        f.write("\n类别说明:\n")
        f.write("  success_multi    成功分离多人 (相似度<0.7 + 两路都有内容)\n")
        f.write("  single_clean     单人输入正确处理 (一路接近静音 + 另一路有内容)\n")
        f.write("  single_no_content 两路都无内容 (录音质量问题)\n")
        f.write("  failed_or_single 高相似度 (相似度>0.92, 崩 or 单人)\n")
        f.write("  ambiguous        中间状态，建议人耳判断\n")
        f.write("\nCSV 字段:\n")
        for fld in fields:
            f.write(f"  {fld}\n")


def _filter_row(row, fields):
    return {k: row.get(k, "") for k in fields}


if __name__ == "__main__":
    main()
