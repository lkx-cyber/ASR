"""
FunASR vs whisper 对比 + 降噪前后对比

每条样本输出 4 个版本的转写：
  whisper-medium 原始 / whisper-medium 降噪
  FunASR Paraformer 原始 / FunASR Paraformer 降噪

关注：
  1. FunASR 是否还有 whisper 那样的"謝謝觀看/阿阿阿"幻觉
  2. 降噪在 FunASR 上的效果（whisper 上是 +/-0.6 字，被幻觉污染）
  3. FunASR 在儿童语音上的实际识别质量
"""
import os
import random
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from denoise import Denoiser, TARGET_SR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")


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


def transcribe_funasr(model, x):
    result = model.generate(input=x, fs=TARGET_SR, batch_size_s=300, disable_pbar=True)
    if not result:
        return ""
    return result[0].get("text", "").strip()


def transcribe_whisper(model, x):
    segments, _ = model.transcribe(x, language="zh", beam_size=1)
    return "".join(s.text for s in segments).strip()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-green", type=int, default=15)
    parser.add_argument("--n-yellow", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 抽样（与 test_denoise_asr 一致的种子，便于直接对照）
    samples = []
    for tier, n in [("green", args.n_green), ("yellow", args.n_yellow)]:
        d = os.path.join(CLEANED_DIR, tier)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".wav")]
        for f in random.sample(files, min(n, len(files))):
            samples.append((tier, os.path.join(d, f)))
    print(f"抽样 {len(samples)} 条\n")

    # 模型加载
    print("加载降噪 + FunASR Paraformer + faster-whisper...")
    dn = Denoiser()
    from funasr import AutoModel
    fa = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
        disable_log=True,
    )
    from faster_whisper import WhisperModel
    wh = WhisperModel("medium", device="cpu", compute_type="int8")
    print("  完成\n")

    print(f"{'#':<3} {'tier':<7} {'file':<14} {'dur':>5}  W原 / W降 / F原 / F降")
    print("-" * 100)

    stats = {"w_orig": [], "w_dn": [], "f_orig": [], "f_dn": []}
    halluc_examples = []

    for i, (tier, path) in enumerate(samples, 1):
        fn = os.path.basename(path).replace(".wav", "")
        x = load_audio(path)
        dur = len(x) / TARGET_SR
        y = dn(x)

        # 4 个转写
        w_o = transcribe_whisper(wh, x)
        w_d = transcribe_whisper(wh, y)
        f_o = transcribe_funasr(fa, x)
        f_d = transcribe_funasr(fa, y)

        stats["w_orig"].append(len(w_o))
        stats["w_dn"].append(len(w_d))
        stats["f_orig"].append(len(f_o))
        stats["f_dn"].append(len(f_d))

        # 检测幻觉模式（whisper 常见）
        halluc_patterns = ["謝謝觀看", "謝謝大家", "下次見", "请订阅", "请点赞",
                           "我爱你", "明镜与点点栏目", "字幕志愿者"]
        is_halluc_w = any(p in w_o or p in w_d for p in halluc_patterns)

        print(f"{i:<3} {tier:<7} {fn[:12]:<14} {dur:>4.1f}s")
        print(f"     W原: {w_o[:60]}")
        print(f"     W降: {w_d[:60]}")
        print(f"     F原: {f_o[:60]}")
        print(f"     F降: {f_d[:60]}")
        if is_halluc_w:
            print(f"     ⚠️ whisper 幻觉")
            halluc_examples.append((fn[:12], w_o, f_o))

    print("-" * 100)
    print("\n" + "=" * 60)
    print("汇总")
    print("=" * 60)
    print(f"{'引擎/版本':<25} {'平均字数':>10} {'非空样本':>10}")
    for k, label in [("w_orig", "whisper 原始"),
                     ("w_dn",   "whisper 降噪"),
                     ("f_orig", "FunASR 原始"),
                     ("f_dn",   "FunASR 降噪")]:
        vals = stats[k]
        nonzero = sum(1 for v in vals if v > 0)
        print(f"{label:<25} {np.mean(vals):>10.1f} {nonzero:>4}/{len(vals)}")

    diff_w = np.array(stats["w_dn"]) - np.array(stats["w_orig"])
    diff_f = np.array(stats["f_dn"]) - np.array(stats["f_orig"])
    print(f"\n降噪带来的字数变化（中位数 / 均值）：")
    print(f"  whisper:  {np.median(diff_w):+.1f} / {np.mean(diff_w):+.2f}")
    print(f"  FunASR:   {np.median(diff_f):+.1f} / {np.mean(diff_f):+.2f}")

    print(f"\n检测到 whisper 幻觉的样本数：{len(halluc_examples)}/{len(samples)}")
    if halluc_examples:
        print("  例子（whisper → FunASR）：")
        for fn, w, f in halluc_examples[:5]:
            print(f"    [{fn}]")
            print(f"      whisper: {w[:60]}")
            print(f"      FunASR:  {f[:60]}")


if __name__ == "__main__":
    main()
