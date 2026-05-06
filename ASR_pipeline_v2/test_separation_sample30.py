"""
分离效果测试脚本 (抽样版)

随机从 recordings_cleaned/green + yellow 抽 N 条（默认 30），
对每条同时跑 baseline (ONNX+Wiener) 和 MossFormer2，
保存所有输出 wav，方便人耳逐个对比。

输出:
  test_separation_sample30/
    {filename}/
      mix.wav                       原始录音
      baseline_轨1.wav               当前模型 + 增强后处理
      baseline_轨2.wav
      mossformer_轨1.wav             MossFormer2 SOTA
      mossformer_轨2.wav
    sample_report.csv               每条的指标和分类
    sample_summary.txt              总体统计

评估维度（自动）:
  - 说话人余弦相似度 (越低越好，<0.7 优秀，>0.92 失败)
  - 能量比 (一极端 = 单人输入；平衡 = 多人输入分得均匀)
  - 两路 ASR 字数 (差距大 = 单人; 都有内容 = 多人成功分离)
  - 自动分类: 成功多人分离 / 单人正确处理 / 疑似失败 / 模糊

人耳评估（必做）:
  在生成的目录里逐个听，判断:
  - 两路是否真的是不同人？
  - 是否有机械感、串音、内容残缺？
  - 跟原 mix 比，分离有没有破坏可懂度？
"""
import os
import sys
import csv
import glob
import random
import time
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
from device_utils import get_device, device_info

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
OUT_DIR = os.path.join(BASE_DIR, "test_separation_sample30")


# ---------- 分离方法 ----------
def separate_baseline(session, raw):
    """当前 ONNX + 增强后处理"""
    x = spectral_denoise(raw)
    x = preemphasis(x)
    x_norm, scale = rms_normalize(x, 0.1)
    in_name = session.get_inputs()[0].name
    outs = session.run(None, {in_name: np.expand_dims(x_norm, 0)})
    sources = [o.squeeze().astype(np.float32) / scale for o in outs]
    sources = [deemphasis(s) for s in sources]
    sources = wiener_refine(raw, sources, 2.0)
    out = []
    for s in sources:
        peak = np.max(np.abs(s))
        if peak > 0.99:
            s = s * (0.99 / peak)
        out.append(s)
    return out


def separate_mossformer(cv_model, raw, tmp_dir):
    """MossFormer2_SS_16K (走临时文件)"""
    os.makedirs(tmp_dir, exist_ok=True)
    in_path = os.path.join(tmp_dir, "_in.wav")
    save_audio(in_path, raw, SR)
    cv_model(input_path=in_path, online_write=True, output_path=tmp_dir)
    inner = os.path.join(tmp_dir, "MossFormer2_SS_16K")
    s1, _ = load_audio(os.path.join(inner, "_in_s1.wav"), verbose=False)
    s2, _ = load_audio(os.path.join(inner, "_in_s2.wav"), verbose=False)
    return [s1, s2]


# ---------- 自动分类 ----------
def classify_separation(sim, energy_ratios, asr_n1, asr_n2):
    """
    返回 (类别, 描述)
    类别:
      success_multi    成功分离多人
      single_clean     单人输入，模型把另一路压低（正确）
      failed_or_single 高相似度，模型没分开 / 输入本就单人
      ambiguous        中间状态
    """
    min_energy = min(energy_ratios)

    # 一路被压到 5% 以下：模型判定单人
    if min_energy < 0.05:
        if max(asr_n1, asr_n2) >= 3:
            return "single_clean", "一路接近静音，单人正确处理"
        return "single_no_content", "一路静音但另一路也无内容"

    # 两路相似度极高：要么模型崩、要么输入本就单人
    if sim > 0.92:
        return "failed_or_single", f"相似度 {sim:.2f} 过高"

    # 两路都有内容 + 相似度低 = 成功多人分离
    if sim < 0.70 and min(asr_n1, asr_n2) >= 3:
        return "success_multi", f"相似度 {sim:.2f}，两路都有内容"

    return "ambiguous", f"相似度 {sim:.2f}，min_energy {min_energy:.2f}"


# ---------- 主流程 ----------
def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=30, help="抽样数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子（同一种子结果可复现）")
    p.add_argument("--tiers", nargs="+", default=["green", "yellow"],
                   help="从哪些桶抽样")
    p.add_argument("--no-mossformer", action="store_true",
                   help="跳过 MossFormer2（节省时间）")
    args = p.parse_args()

    random.seed(args.seed)

    # 收集候选
    candidates = []
    for tier in args.tiers:
        files = sorted(glob.glob(os.path.join(CLEANED_DIR, tier, "*.wav")))
        for f in files:
            candidates.append((tier, f))
    if not candidates:
        print(f"未在 {CLEANED_DIR}/[{','.join(args.tiers)}]/ 找到 wav")
        return

    print(f"候选池: {len(candidates)} 条 (来自 {args.tiers})")

    # 抽样
    sampled = random.sample(candidates, min(args.n, len(candidates)))
    print(f"抽样: {len(sampled)} 条 (seed={args.seed})\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    tmp_dir = os.path.join(OUT_DIR, "_tmp_mossformer")

    device = get_device()
    print(f"🖥️  Device: {device_info()}")

    # 加载模型
    print("\n加载 ONNX baseline 分离模型 (CPU, 按方案 A)...")
    session = ort.InferenceSession(MODEL_PATH)  # baseline 仍用 CPU

    cv = None
    if not args.no_mossformer:
        print(f"加载 MossFormer2_SS_16K ({device})...")
        from clearvoice import ClearVoice
        # ClearVoice 内部用 torch，会自动检测 cuda
        cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])

    print(f"加载 FunASR ({device})...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
        device=device,
    )

    def transcribe(samples):
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
    csv_path = os.path.join(OUT_DIR, "sample_report.csv")
    fields = [
        "filename", "tier", "duration",
        "baseline_sim", "baseline_e1", "baseline_e2",
        "baseline_asr1", "baseline_asr2", "baseline_n1", "baseline_n2",
        "baseline_class", "baseline_class_reason",
        "mf2_sim", "mf2_e1", "mf2_e2",
        "mf2_asr1", "mf2_asr2", "mf2_n1", "mf2_n2",
        "mf2_class", "mf2_class_reason",
        "mix_asr",
    ]
    csv_file = open(csv_path, "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=fields)
    writer.writeheader()

    summary = {"baseline": {}, "mf2": {}}
    print("\n开始测试...\n")
    t_start = time.time()

    for idx, (tier, path) in enumerate(sampled, 1):
        fname = os.path.splitext(os.path.basename(path))[0]
        print(f"[{idx:>2}/{len(sampled)}] {tier:<6} {fname}")

        item_dir = os.path.join(OUT_DIR, fname)
        os.makedirs(item_dir, exist_ok=True)

        # 加载原音频
        try:
            mix, sr = load_audio(path, verbose=False)
        except Exception as e:
            print(f"  ⚠️  加载失败: {e}")
            continue

        save_audio(os.path.join(item_dir, "mix.wav"), mix, SR)
        mix_text, _ = transcribe(mix)

        row = {
            "filename": fname, "tier": tier,
            "duration": round(len(mix) / SR, 2),
            "mix_asr": mix_text,
        }

        # === Baseline ===
        try:
            b_sources = separate_baseline(session, mix)
            b_metrics = evaluate_separation(mix, b_sources)
            b_sim = speaker_similarity(b_sources)
            b_t1, b_n1 = transcribe(b_sources[0])
            b_t2, b_n2 = transcribe(b_sources[1])
            b_class, b_reason = classify_separation(
                b_sim if isinstance(b_sim, float) else 1.0,
                b_metrics["能量比"], b_n1, b_n2
            )

            save_audio(os.path.join(item_dir, "baseline_轨1.wav"), b_sources[0], SR)
            save_audio(os.path.join(item_dir, "baseline_轨2.wav"), b_sources[1], SR)

            row.update({
                "baseline_sim": round(b_sim, 3) if isinstance(b_sim, float) else "n/a",
                "baseline_e1": b_metrics["能量比"][0],
                "baseline_e2": b_metrics["能量比"][1],
                "baseline_asr1": b_t1, "baseline_asr2": b_t2,
                "baseline_n1": b_n1, "baseline_n2": b_n2,
                "baseline_class": b_class, "baseline_class_reason": b_reason,
            })
            summary["baseline"][b_class] = summary["baseline"].get(b_class, 0) + 1
            print(f"    baseline: sim={b_sim if isinstance(b_sim,float) else 'n/a'}  "
                  f"能量={b_metrics['能量比']}  → {b_class}")
        except Exception as e:
            print(f"  ⚠️  baseline 失败: {e}")
            row.update({"baseline_class": f"error: {e}"})

        # === MossFormer2 ===
        if cv is not None:
            try:
                m_sources = separate_mossformer(cv, mix, tmp_dir)
                m_metrics = evaluate_separation(mix, m_sources)
                m_sim = speaker_similarity(m_sources)
                m_t1, m_n1 = transcribe(m_sources[0])
                m_t2, m_n2 = transcribe(m_sources[1])
                m_class, m_reason = classify_separation(
                    m_sim if isinstance(m_sim, float) else 1.0,
                    m_metrics["能量比"], m_n1, m_n2
                )

                save_audio(os.path.join(item_dir, "mossformer_轨1.wav"), m_sources[0], SR)
                save_audio(os.path.join(item_dir, "mossformer_轨2.wav"), m_sources[1], SR)

                row.update({
                    "mf2_sim": round(m_sim, 3) if isinstance(m_sim, float) else "n/a",
                    "mf2_e1": m_metrics["能量比"][0],
                    "mf2_e2": m_metrics["能量比"][1],
                    "mf2_asr1": m_t1, "mf2_asr2": m_t2,
                    "mf2_n1": m_n1, "mf2_n2": m_n2,
                    "mf2_class": m_class, "mf2_class_reason": m_reason,
                })
                summary["mf2"][m_class] = summary["mf2"].get(m_class, 0) + 1
                print(f"    mossformer: sim={m_sim if isinstance(m_sim,float) else 'n/a'}  "
                      f"能量={m_metrics['能量比']}  → {m_class}")
            except Exception as e:
                print(f"  ⚠️  mossformer 失败: {e}")
                row.update({"mf2_class": f"error: {e}"})

        writer.writerow(row)
        csv_file.flush()
        print()

    csv_file.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # 汇总
    elapsed = (time.time() - t_start) / 60
    print(f"\n{'='*78}")
    print(f"✓ 测试完成，耗时 {elapsed:.1f} 分钟")
    print(f"\n📊 自动分类汇总 (baseline):")
    for cls, count in sorted(summary["baseline"].items(), key=lambda x: -x[1]):
        print(f"  {cls:<22} {count}")
    if cv is not None:
        print(f"\n📊 自动分类汇总 (MossFormer2):")
        for cls, count in sorted(summary["mf2"].items(), key=lambda x: -x[1]):
            print(f"  {cls:<22} {count}")

    print(f"\n📁 输出位置: {OUT_DIR}/")
    print(f"   - {len(sampled)} 个子目录，每个含 mix.wav + 4 个分离 wav (baseline×2 + mossformer×2)")
    print(f"   - sample_report.csv  全部指标 + 自动分类")
    print(f"\n👂 下一步: 用 Audacity 或 VSCode Audio Preview 听各子目录里的 wav，")
    print(f"   验证自动分类是否符合人耳判断。")

    # 写 summary.txt
    with open(os.path.join(OUT_DIR, "sample_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"分离效果测试 - 抽样 {len(sampled)} 条\n")
        f.write(f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"种子: {args.seed}\n")
        f.write(f"桶: {args.tiers}\n")
        f.write(f"耗时: {elapsed:.1f} 分钟\n\n")
        f.write("Baseline 自动分类:\n")
        for cls, count in sorted(summary["baseline"].items(), key=lambda x: -x[1]):
            f.write(f"  {cls}: {count}\n")
        if cv is not None:
            f.write("\nMossFormer2 自动分类:\n")
            for cls, count in sorted(summary["mf2"].items(), key=lambda x: -x[1]):
                f.write(f"  {cls}: {count}\n")
        f.write("\n类别说明:\n")
        f.write("  success_multi    成功分离多人 (相似度<0.7 + 两路都有内容)\n")
        f.write("  single_clean     单人输入正确处理 (一路接近静音 + 另一路有内容)\n")
        f.write("  single_no_content 一路静音但另一路也无内容 (不太对)\n")
        f.write("  failed_or_single 高相似度 (相似度>0.92, 模型崩 or 输入单人)\n")
        f.write("  ambiguous        中间状态，建议人耳判断\n")


if __name__ == "__main__":
    main()
