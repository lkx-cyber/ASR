"""
ASR 双引擎对比：faster-whisper vs FunASR Paraformer-zh

测试集：v1 项目里的 2in1.wav 和分离结果（4 秒 2 人混合，已知 ground truth）

注意：此脚本依赖 sibling 目录 ../公司项目/audiosep/，
      只在你保留了 v1 仓库（MasterQiann/ASR）的同级 clone 时能跑。
      如果只 clone 了 ASR_pipeline_v2，这个脚本会报"找不到文件"。
"""
import os
import sys
import time
import re
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio

OLD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "audiosep"))

REF1 = "我考试考得太差了我都有点不想活了"
REF2 = "我养了一只小狗汪汪汪汪汪"

AUDIOS = [
    ("混合音频",       os.path.join(OLD_DIR, "2in1.wav")),
    ("原版-轨1",       os.path.join(OLD_DIR, "分离_说话人_1.wav")),
    ("原版-轨2",       os.path.join(OLD_DIR, "分离_说话人_2.wav")),
    ("增强版-轨1",     os.path.join(OLD_DIR, "分离_增强_说话人_1.wav")),
    ("增强版-轨2",     os.path.join(OLD_DIR, "分离_增强_说话人_2.wav")),
]


# ---------- CER ----------
_t2s = None
def to_simplified(s):
    global _t2s
    if _t2s is None:
        from opencc import OpenCC
        _t2s = OpenCC("t2s")
    return _t2s.convert(s)


def normalize(s):
    return re.sub(r"[^\w一-鿿]", "", to_simplified(s))


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (dp[i - 1][j - 1] if a[i - 1] == b[j - 1]
                       else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]))
    return dp[n][m]


def cer(hyp, ref):
    h, r = normalize(hyp), normalize(ref)
    if not r:
        return 1.0 if h else 0.0
    return edit_distance(h, r) / len(r)


def best_cer(hyp, refs):
    return min(cer(hyp, r) for r in refs)


# ---------- A. faster-whisper ----------
def build_whisper():
    from faster_whisper import WhisperModel
    print("加载 faster-whisper medium...")
    return WhisperModel("medium", device="cpu", compute_type="int8")


def whisper_transcribe(model, path):
    samples, sr = load_audio(path, verbose=False)
    t0 = time.time()
    segs, _ = model.transcribe(samples, language="zh", beam_size=5,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=300))
    text = "".join(s.text for s in segs).strip()
    return text, time.time() - t0


# ---------- B. FunASR ----------
def build_funasr():
    from funasr import AutoModel
    print("加载 FunASR (Paraformer-zh + cam++ + ct-punc)...")
    return AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", spk_model="cam++",
        disable_update=True, disable_log=True, disable_pbar=True,
    )


def funasr_transcribe(model, path):
    samples, sr = load_audio(path, verbose=False)
    t0 = time.time()
    result = model.generate(input=samples, fs=sr, batch_size_s=300,
                             return_spk_res=True)
    elapsed = time.time() - t0
    # 聚合所有 segment 的 text（不区分 spk）—— 单纯比 ASR 内容
    parts = []
    for item in result:
        for sent in item.get("sentence_info", []):
            parts.append(sent.get("text", "").strip())
        if not item.get("sentence_info"):
            parts.append(item.get("text", "").strip())
    text = "".join(parts).strip()
    return text, elapsed


# ---------- 主流程 ----------
def main():
    refs = [REF1, REF2]
    print(f"参考 1: {REF1}")
    print(f"参考 2: {REF2}\n")

    print("=" * 90)
    whisper = build_whisper()
    funasr = build_funasr()
    print()

    # 表头
    print(f"{'音频':<14}|  {'whisper-medium CER':<22}|  {'FunASR CER':<22}|  最佳")
    print("-" * 90)

    rows = []
    for label, path in AUDIOS:
        if not os.path.exists(path):
            print(f"{label:<14}|  (未找到 {path})")
            continue

        wt, wd = whisper_transcribe(whisper, path)
        ft, fd = funasr_transcribe(funasr, path)

        wcer = best_cer(wt, refs)
        fcer = best_cer(ft, refs)
        winner = "FunASR" if fcer < wcer else ("whisper" if wcer < fcer else "tie")

        rows.append((label, wt, wd, wcer, ft, fd, fcer, winner))
        print(f"{label:<14}|  {wcer*100:>5.1f}%  ({wd:4.1f}s)         |  "
              f"{fcer*100:>5.1f}%  ({fd:4.1f}s)         |  {winner}")

    print("\n=== 详细转写 ===")
    for label, wt, wd, wcer, ft, fd, fcer, winner in rows:
        print(f"\n[{label}]")
        print(f"  whisper-medium: {wt}")
        print(f"  FunASR:         {ft}")


if __name__ == "__main__":
    main()
