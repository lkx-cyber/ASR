"""
基于 groundtruth.csv 评估候选 pipeline 的 CER。

特性：
  - 多 pipeline 对比（注册表，易扩展）
  - 文本归一化（去标点 / 全半角 / 繁简 / 大小写）
  - 输出：每条 detail + 整体 summary + pipeline 对比表
  - 支持只跑某些 pipeline（避免每次重测）

用法：
  python eval_with_groundtruth.py                       # 默认全部
  python eval_with_groundtruth.py --pipelines P1 P2     # 只跑指定
  python eval_with_groundtruth.py --gt groundtruth.csv  # 自定义 GT
  python eval_with_groundtruth.py --audio-dir recordings_cleaned_v2/green++

输出（eval_results/<timestamp>/）：
  summary.txt   总览 + 各 pipeline 对比 + 错误模式分布
  detail.csv    每条样本逐条结果（含 hyp / ref / norm / edit_distance / cer）
"""
import argparse
import csv
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


# ============== 文本归一化 ==============
import re
import string

try:
    from opencc import OpenCC
    _CC = OpenCC("t2s")  # 繁→简
except Exception:
    _CC = None

# 中英文标点全集
_PUNCT = set(
    "，。！？、；：""''（）【】《》「」『』〈〉…—·～￥"
    + string.punctuation
    + " \t\n\r"
)


def normalize_text(s: str) -> str:
    """
    文本归一化（用于 CER 计算）：
      1. 繁体 → 简体
      2. 全角 → 半角
      3. 大小写统一（小写）
      4. 去除所有标点 + 空白
    """
    if not s:
        return ""
    if _CC is not None:
        s = _CC.convert(s)
    # 全角→半角
    out = []
    for ch in s:
        code = ord(ch)
        if code == 0x3000:  # 全角空格
            ch = " "
        elif 0xFF01 <= code <= 0xFF5E:
            ch = chr(code - 0xFEE0)
        out.append(ch)
    s = "".join(out).lower()
    # 去标点+空白
    s = "".join(ch for ch in s if ch not in _PUNCT)
    return s


# ============== CER ==============
def edit_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                cur[j] = prev[j - 1]
            else:
                cur[j] = 1 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    return prev[n]


def cer(ref: str, hyp: str) -> tuple:
    """返回 (edit_distance, ref_len, cer_value)，使用归一化文本。"""
    r = normalize_text(ref)
    h = normalize_text(hyp)
    ed = edit_distance(r, h)
    rl = len(r) if r else 1
    return ed, len(r), ed / rl


# ============== Pipeline 注册表 ==============
# 每个 pipeline 是一个 callable: (audio: np.ndarray, sr: int) -> str
# 在 main() 里实例化模型后注册

PIPELINES = {}


def register(name):
    def deco(fn):
        PIPELINES[name] = fn
        return fn
    return deco


# ============== 主流程 ==============
def load_audio(path, target_sr=16000):
    """兼容 MP4-in-WAV 容器的加载"""
    import soundfile as sf
    try:
        x, sr = sf.read(path, dtype="float32")
    except Exception:
        import librosa
        x, sr = librosa.load(path, sr=None, mono=True)
        x = x.astype(np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != target_sr:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=target_sr).astype(np.float32)
    return x, target_sr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="groundtruth.csv",
                        help="ground truth CSV 路径（相对项目根或绝对）")
    parser.add_argument("--audio-dir", default="recordings_cleaned_v2/green++",
                        help="音频目录")
    parser.add_argument("--pipelines", nargs="+", default=None,
                        help="只跑指定 pipeline（默认全部）")
    parser.add_argument("--out-root", default="eval_results",
                        help="输出根目录")
    parser.add_argument("--limit", type=int, default=None,
                        help="只评估前 N 条（调试用）")
    args = parser.parse_args()

    gt_path = args.gt if os.path.isabs(args.gt) else os.path.join(BASE_DIR, args.gt)
    audio_dir = args.audio_dir if os.path.isabs(args.audio_dir) else os.path.join(BASE_DIR, args.audio_dir)
    out_root = args.out_root if os.path.isabs(args.out_root) else os.path.join(BASE_DIR, args.out_root)

    if not os.path.exists(gt_path):
        print(f"❌ GT 不存在: {gt_path}")
        sys.exit(1)
    if not os.path.isdir(audio_dir):
        print(f"❌ 音频目录不存在: {audio_dir}")
        sys.exit(1)

    # 读 GT
    with open(gt_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    valid_rows = [r for r in rows if r["correct_text"].strip() and r["correct_text"].strip() != "/n"]
    if args.limit:
        valid_rows = valid_rows[: args.limit]
    print(f"📚 GT: {len(rows)} 条原始, {len(valid_rows)} 条有效（去掉 /n 跳过条）")

    # ============== 注册 pipeline ==============
    print("加载模型...")
    from funasr import AutoModel
    fa = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    @register("P1_raw_funasr")
    def p1(audio, sr):
        """当前生产架构：raw → FunASR Paraformer"""
        r = fa.generate(input=audio, fs=sr, batch_size_s=300, disable_pbar=True)
        return r[0].get("text", "").strip() if r else ""

    # 可选：DFN3 降噪 → FunASR（轻量加进来作为对照）
    try:
        from denoise import Denoiser
        _dn = Denoiser()

        @register("P2_dfn3_funasr")
        def p2(audio, sr):
            """前端 DFN3 降噪 → FunASR"""
            denoised = _dn(audio)
            r = fa.generate(input=denoised, fs=sr, batch_size_s=300, disable_pbar=True)
            return r[0].get("text", "").strip() if r else ""
    except Exception as e:
        print(f"⚠️ 降噪 pipeline 注册失败（跳过）: {e}")

    # P3/P4: 录音开头补静音 padding（修开头吞字）
    @register("P3_pad200ms_funasr")
    def p3(audio, sr):
        """开头加 200ms 静音 → FunASR"""
        padding = np.zeros(int(0.2 * sr), dtype=np.float32)
        padded = np.concatenate([padding, audio])
        r = fa.generate(input=padded, fs=sr, batch_size_s=300, disable_pbar=True)
        return r[0].get("text", "").strip() if r else ""

    @register("P4_pad500ms_funasr")
    def p4(audio, sr):
        """开头加 500ms 静音 → FunASR"""
        padding = np.zeros(int(0.5 * sr), dtype=np.float32)
        padded = np.concatenate([padding, audio])
        r = fa.generate(input=padded, fs=sr, batch_size_s=300, disable_pbar=True)
        return r[0].get("text", "").strip() if r else ""

    # 过滤要跑的
    if args.pipelines:
        keep = set(args.pipelines)
        unknown = keep - set(PIPELINES)
        if unknown:
            print(f"❌ 未知 pipeline: {unknown}")
            print(f"   可用: {list(PIPELINES.keys())}")
            sys.exit(1)
        run_names = [n for n in PIPELINES if n in keep]
    else:
        run_names = list(PIPELINES.keys())

    print(f"📋 将评估 {len(run_names)} 个 pipeline: {run_names}\n")

    # ============== 跑评估 ==============
    # results[pipeline][filename] = {ref, hyp, ed, ref_len, cer, secs}
    results = {n: {} for n in run_names}

    for i, row in enumerate(valid_rows, 1):
        fname = row["filename"]
        ref = row["correct_text"].strip()
        path = os.path.join(audio_dir, fname)
        if not os.path.exists(path):
            print(f"⚠️ [{i}/{len(valid_rows)}] {fname} 不存在，跳过")
            continue
        try:
            audio, sr = load_audio(path)
        except Exception as e:
            print(f"⚠️ [{i}/{len(valid_rows)}] {fname} 加载失败: {e}")
            continue

        # 对每个 pipeline 跑
        line_parts = [f"[{i}/{len(valid_rows)}] {fname[:12]}"]
        for name in run_names:
            t0 = time.time()
            try:
                hyp = PIPELINES[name](audio, sr)
            except Exception as e:
                hyp = f"<err:{e}>"
            secs = time.time() - t0
            ed, rl, c = cer(ref, hyp)
            results[name][fname] = {
                "ref": ref, "hyp": hyp, "ed": ed, "ref_len": rl, "cer": c, "secs": secs
            }
            line_parts.append(f"{name}: cer={c*100:.1f}% ({secs:.1f}s)")
        print("  | ".join(line_parts))

    # ============== 输出 ==============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(out_root, timestamp)
    os.makedirs(out_dir, exist_ok=True)

    # detail.csv
    detail_path = os.path.join(out_dir, "detail.csv")
    fields = ["filename", "ref"]
    for name in run_names:
        fields += [f"{name}_hyp", f"{name}_ed", f"{name}_cer", f"{name}_secs"]
    with open(detail_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        # 取所有出现过的 filename
        all_files = sorted({fn for n in run_names for fn in results[n]})
        for fn in all_files:
            row_out = {"filename": fn}
            ref = next((results[n][fn]["ref"] for n in run_names if fn in results[n]), "")
            row_out["ref"] = ref
            for name in run_names:
                r = results[name].get(fn)
                if r:
                    row_out[f"{name}_hyp"] = r["hyp"]
                    row_out[f"{name}_ed"] = r["ed"]
                    row_out[f"{name}_cer"] = round(r["cer"] * 100, 2)
                    row_out[f"{name}_secs"] = round(r["secs"], 2)
            w.writerow(row_out)

    # summary.txt
    summary_path = os.path.join(out_dir, "summary.txt")
    lines = []
    lines.append("=" * 72)
    lines.append(f"评估时间: {timestamp}")
    lines.append(f"GT 文件: {gt_path}")
    lines.append(f"音频目录: {audio_dir}")
    lines.append(f"有效样本: {len(valid_rows)}")
    lines.append(f"参与 pipeline: {run_names}")
    lines.append("")

    # 各 pipeline 总指标
    lines.append("=" * 72)
    lines.append("【整体指标】")
    lines.append("")
    lines.append(f"{'Pipeline':<22} {'整体CER':>10} {'平均CER':>10} {'完美率':>10} "
                 f"{'<5%':>6} {'<10%':>6} {'<20%':>6} {'>=20%':>6} {'平均耗时':>10}")
    lines.append("-" * 100)

    perf_summary = {}
    for name in run_names:
        items = list(results[name].values())
        if not items:
            continue
        total_ed = sum(r["ed"] for r in items)
        total_ref = sum(r["ref_len"] for r in items)
        overall = total_ed / total_ref if total_ref else 0
        avg_cer = np.mean([r["cer"] for r in items])
        perfect = sum(1 for r in items if r["ed"] == 0)
        cer_lt5 = sum(1 for r in items if r["cer"] < 0.05)
        cer_lt10 = sum(1 for r in items if r["cer"] < 0.10)
        cer_lt20 = sum(1 for r in items if r["cer"] < 0.20)
        cer_ge20 = sum(1 for r in items if r["cer"] >= 0.20)
        avg_secs = np.mean([r["secs"] for r in items])
        n = len(items)
        lines.append(
            f"{name:<22} "
            f"{overall*100:>9.2f}% "
            f"{avg_cer*100:>9.2f}% "
            f"{perfect:>4}/{n:<4} "
            f"{cer_lt5:>6} "
            f"{cer_lt10:>6} "
            f"{cer_lt20:>6} "
            f"{cer_ge20:>6} "
            f"{avg_secs:>9.2f}s"
        )
        perf_summary[name] = (overall, avg_cer, perfect, n)

    # pipeline 对比（如果有 ≥2 个）
    if len(run_names) >= 2:
        lines.append("")
        lines.append("=" * 72)
        lines.append("【Pipeline 对比】（vs 第一个 pipeline 作为 baseline）")
        lines.append("")
        baseline = run_names[0]
        baseline_files = set(results[baseline])
        for name in run_names[1:]:
            shared = baseline_files & set(results[name])
            if not shared:
                continue
            improved = same = degraded = 0
            ed_diff = []
            for fn in shared:
                b = results[baseline][fn]
                c = results[name][fn]
                if c["ed"] < b["ed"]:
                    improved += 1
                elif c["ed"] > b["ed"]:
                    degraded += 1
                else:
                    same += 1
                ed_diff.append(c["ed"] - b["ed"])
            lines.append(f"  {name} vs {baseline}（共 {len(shared)} 条）:")
            lines.append(f"    改善: {improved} 条")
            lines.append(f"    持平: {same} 条")
            lines.append(f"    退化: {degraded} 条")
            lines.append(f"    总编辑距离差值: {sum(ed_diff):+d}")
            lines.append("")

    # Top-5 最难 case（用第一个 pipeline 排）
    if run_names:
        lines.append("=" * 72)
        lines.append(f"【最难识别的 5 条】（按 {run_names[0]} 编辑距离排）")
        lines.append("")
        items = sorted(
            results[run_names[0]].items(),
            key=lambda kv: -kv[1]["ed"]
        )[:5]
        for fn, r in items:
            lines.append(f"  {fn[:16]}  ed={r['ed']}, cer={r['cer']*100:.1f}%")
            lines.append(f"     ref: {r['ref']}")
            lines.append(f"     hyp: {r['hyp']}")
            lines.append("")

    summary = "\n".join(lines)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)

    print()
    print(summary)
    print()
    print(f"✓ 详情: {detail_path}")
    print(f"✓ 总结: {summary_path}")


if __name__ == "__main__":
    main()
