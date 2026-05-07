"""
横向对比多套 pipeline 的 ASR 输出，不修改 baseline 代码。

对比的 6 套 pipeline（同一份输入，6 路并跑）：

  P1  raw → FunASR                              （最简，无任何前处理）
  P2  DFN3降噪 → FunASR                          （前端只加降噪）
  P3  baseline分离(谱减+ONNX+Wiener) → FunASR    （现有 baseline 流程）
  P4  DFN3 → baseline分离 → FunASR               （前端降噪 + 现有分离）
  P5  baseline分离 → whisper                     （baseline 早期形态，作历史参照）
  P6  DFN3 → FunASR (无分离)                     ← 已确认最简但最稳的候选

对每条样本输出 4 个文本结果（分离的取两轨）：
  - 不分离的 pipeline: 单一文本
  - 分离的 pipeline: track1 / track2 / mix 三个文本

输出：
  - 控制台对照
  - test_pipeline_compare/{filename}/  各 pipeline 中间产物 wav，供听感校验
  - test_pipeline_compare/report.csv    便于数据分析
"""
import argparse
import csv
import os
import random
import sys
import time
import warnings

import numpy as np
import soundfile as sf
import librosa

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# 复用 baseline 的分离逻辑，不改它
from separate_baseline import separate_one, MODEL_PATH, SR
import onnxruntime as ort

# DFN3 降噪
from denoise import Denoiser

CLEANED_DIR = os.path.join(BASE_DIR, "recordings_cleaned")
OUT_DIR = os.path.join(BASE_DIR, "test_pipeline_compare")


def load_audio(path):
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        x, sr = librosa.load(path, sr=None, mono=True)
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        x = librosa.resample(x, orig_sr=sr, target_sr=SR).astype(np.float32)
    return x


# ---------- 收集样本 ----------
def pick_samples(n_green, n_yellow, seed):
    random.seed(seed)
    out = []
    for tier, n in [("green", n_green), ("yellow", n_yellow)]:
        d = os.path.join(CLEANED_DIR, tier)
        if not os.path.isdir(d):
            continue
        files = [f for f in os.listdir(d) if f.endswith(".wav")]
        for f in random.sample(files, min(n, len(files))):
            out.append((tier, os.path.join(d, f)))
    return out


# ---------- 跑一个文件的所有 pipeline ----------
def run_all_pipelines(path, sep_session, denoiser, fa, wh, save_dir=None):
    """返回 dict: pipeline_id → result（文本 + 时长 + 备注）"""
    name = os.path.basename(path).replace(".wav", "")[:14]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    raw = load_audio(path)
    if save_dir:
        sf.write(os.path.join(save_dir, "00_raw.wav"), raw, SR)

    def fa_text(x):
        try:
            r = fa.generate(input=x, fs=SR, batch_size_s=300, disable_pbar=True)
            return r[0].get("text", "").strip() if r else ""
        except Exception as e:
            return f"<err:{e}>"

    def wh_text(x):
        try:
            seg, _ = wh.transcribe(x, language="zh", beam_size=1)
            return "".join(s.text for s in seg).strip()
        except Exception as e:
            return f"<err:{e}>"

    results = {}

    # P1: raw → FunASR
    t = time.time()
    txt = fa_text(raw)
    results["P1_raw→FunASR"] = {"text": txt, "secs": time.time() - t}

    # P2: DFN3 → FunASR
    t = time.time()
    dn_audio = denoiser(raw)
    if save_dir:
        sf.write(os.path.join(save_dir, "02_dfn3.wav"), dn_audio, SR)
    txt = fa_text(dn_audio)
    results["P2_DFN3→FunASR"] = {"text": txt, "secs": time.time() - t}

    # P3: baseline 分离 → FunASR
    t = time.time()
    tracks = separate_one(sep_session, raw)
    if save_dir:
        for i, s in enumerate(tracks):
            sf.write(os.path.join(save_dir, f"03_baseline_轨{i+1}.wav"), s, SR)
    t1 = fa_text(tracks[0])
    t2 = fa_text(tracks[1])
    tmix = fa_text(raw)  # 复用 raw 的转写当 mix（与 P1 重合）
    results["P3_baseline→FunASR"] = {
        "text": f"轨1: {t1} | 轨2: {t2}",
        "track1": t1, "track2": t2, "secs": time.time() - t,
    }

    # P4: DFN3 → baseline 分离 → FunASR
    t = time.time()
    tracks_dn = separate_one(sep_session, dn_audio)
    if save_dir:
        for i, s in enumerate(tracks_dn):
            sf.write(os.path.join(save_dir, f"04_dfn3_baseline_轨{i+1}.wav"), s, SR)
    t1d = fa_text(tracks_dn[0])
    t2d = fa_text(tracks_dn[1])
    results["P4_DFN3→baseline→FunASR"] = {
        "text": f"轨1: {t1d} | 轨2: {t2d}",
        "track1": t1d, "track2": t2d, "secs": time.time() - t,
    }

    # P5: baseline 分离 → whisper （历史对照）
    t = time.time()
    t1w = wh_text(tracks[0])
    t2w = wh_text(tracks[1])
    results["P5_baseline→whisper"] = {
        "text": f"轨1: {t1w} | 轨2: {t2w}",
        "track1": t1w, "track2": t2w, "secs": time.time() - t,
    }

    # P6: DFN3 → FunASR（与 P2 等价，已包含在 P2）—— 跳过

    # raw → whisper （也作历史对照）
    t = time.time()
    rwh = wh_text(raw)
    results["P0_raw→whisper"] = {"text": rwh, "secs": time.time() - t}

    return name, results, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-green", type=int, default=10)
    parser.add_argument("--n-yellow", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-save-wavs", action="store_true")
    args = parser.parse_args()

    samples = pick_samples(args.n_green, args.n_yellow, args.seed)
    print(f"抽样 {len(samples)} 条（green={args.n_green}, yellow={args.n_yellow}）\n")

    print("加载所有模型...")
    sep_session = ort.InferenceSession(MODEL_PATH)
    denoiser = Denoiser()
    from funasr import AutoModel
    fa = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", disable_update=True, disable_log=True,
    )
    from faster_whisper import WhisperModel
    wh = WhisperModel("medium", device="cpu", compute_type="int8")
    print("  完成\n")

    csv_rows = []
    for tier, path in samples:
        save_dir = None
        if not args.no_save_wavs:
            save_dir = os.path.join(OUT_DIR, f"{tier}_{os.path.basename(path)[:14]}")
        name, results, raw = run_all_pipelines(
            path, sep_session, denoiser, fa, wh, save_dir
        )
        dur = len(raw) / SR

        print("=" * 90)
        print(f"📁 [{tier}] {name}  ({dur:.1f}s)")
        for pid, r in results.items():
            print(f"  {pid:<28} ({r['secs']:>4.1f}s)  {r['text'][:70]}")
        print()

        # CSV 记录
        row = {"tier": tier, "file": name, "dur": f"{dur:.2f}"}
        for pid, r in results.items():
            row[pid] = r["text"]
            row[f"{pid}_secs"] = f"{r['secs']:.2f}"
        csv_rows.append(row)

    # 保存 CSV
    if csv_rows:
        os.makedirs(OUT_DIR, exist_ok=True)
        csv_path = os.path.join(OUT_DIR, "report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            w.writeheader()
            w.writerows(csv_rows)
        print(f"\n✓ CSV 报告: {csv_path}")
        if save_dir:
            print(f"✓ 各 pipeline 中间 wav: {OUT_DIR}/")


if __name__ == "__main__":
    main()
