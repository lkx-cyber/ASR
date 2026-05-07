"""
降噪前后 ASR 对比测试

抽样 N 条录音，对原始/降噪两个版本分别用 faster-whisper 转写，
统计文本一致性变化、字数差异等。

注：没有人工标注的真值（ground truth），所以无法算 CER。这里看的是：
  - 降噪后是否仍然能转出文字（是否过度抑制）
  - 文本是否变化（是否引入错误 / 修复幻觉）
  - 字数变化
"""
import argparse
import os
import random
import sys
import time
import warnings

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

# 确保 denoise 模块在 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise import Denoiser, TARGET_SR

import librosa

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
OUT_DIR = os.path.join(BASE_DIR, "test_denoise_asr")


def load_audio(path):
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        x, sr = librosa.load(path, sr=None, mono=True)
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != TARGET_SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)
    return x


def transcribe(model, audio):
    segments, _ = model.transcribe(audio, language="zh", beam_size=1)
    return "".join(s.text for s in segments).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-green", type=int, default=15)
    parser.add_argument("--n-yellow", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--whisper-size", default="medium",
                        choices=["tiny", "base", "small", "medium", "large-v3"])
    parser.add_argument("--save-wavs", action="store_true",
                        help="保存原始/降噪 wav 供听感复核")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.save_wavs:
        os.makedirs(OUT_DIR, exist_ok=True)

    # 抽样
    samples = []
    for tier, n in [("green", args.n_green), ("yellow", args.n_yellow)]:
        d = os.path.join(CLEANED_DIR, tier)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".wav")]
        for f in random.sample(files, min(n, len(files))):
            samples.append((tier, os.path.join(d, f)))
    print(f"抽样 {len(samples)} 条（green={args.n_green}, yellow={args.n_yellow}）\n")

    # 加载模型
    print("加载 DFN3 + faster-whisper...")
    dn = Denoiser()
    from faster_whisper import WhisperModel
    asr = WhisperModel(args.whisper_size, device="cpu", compute_type="int8")
    print("  完成\n")

    # 处理
    print(f"{'tier':<7} {'file':<14} {'dur':>5}  原始 ASR / 降噪 ASR")
    print("-" * 100)
    stats = {"orig_chars": [], "dn_chars": [], "diff_chars": [], "same_text": 0}
    for tier, path in samples:
        fn = os.path.basename(path).replace(".wav", "")
        x = load_audio(path)
        dur = len(x) / TARGET_SR

        y = dn(x)

        t1 = time.time()
        text_orig = transcribe(asr, x)
        text_dn = transcribe(asr, y)
        # 静默以避免太长

        co, cd = len(text_orig), len(text_dn)
        stats["orig_chars"].append(co)
        stats["dn_chars"].append(cd)
        stats["diff_chars"].append(cd - co)
        if text_orig == text_dn:
            stats["same_text"] += 1

        print(f"{tier:<7} {fn[:12]:<14} {dur:>4.1f}s")
        print(f"  原始: {text_orig[:80]}")
        print(f"  降噪: {text_dn[:80]}")

        if args.save_wavs:
            sub = os.path.join(OUT_DIR, f"{tier}_{fn[:12]}")
            os.makedirs(sub, exist_ok=True)
            sf.write(os.path.join(sub, "1_原始.wav"), x, TARGET_SR)
            sf.write(os.path.join(sub, "2_降噪.wav"), y, TARGET_SR)

    print("-" * 100)
    print(f"\n汇总（{len(samples)} 条）：")
    print(f"  原始 平均字数：{np.mean(stats['orig_chars']):.1f}")
    print(f"  降噪 平均字数：{np.mean(stats['dn_chars']):.1f}")
    print(f"  字数变化：均值 {np.mean(stats['diff_chars']):+.1f}，"
          f"中位数 {np.median(stats['diff_chars']):+.1f}")
    print(f"  转写完全一致：{stats['same_text']}/{len(samples)} "
          f"({100*stats['same_text']/len(samples):.0f}%)")
    print(f"  降噪后仍有内容：{sum(1 for c in stats['dn_chars'] if c > 0)}/{len(samples)}")
    print(f"  降噪后变空文本：{sum(1 for o, d in zip(stats['orig_chars'], stats['dn_chars']) if o > 0 and d == 0)}")

    if args.save_wavs:
        print(f"\n样本 wav 已保存到: {OUT_DIR}")


if __name__ == "__main__":
    main()
