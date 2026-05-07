"""
查找"真有展示价值"的分离样本

策略:
  1. 在 verify_multi/ (已确认多说话人) 中搜
  2. 跑 baseline 分离 + ASR
  3. 评分:
     +3 两路都有 ≥3 字 ASR 内容 (说明真分出两个人)
     +2 两路 ASR 文本不同 (内容 ≠ 重复)
     +2 两路说话人余弦相似度 < 0.7 (真不同的人)
     +1 主轨 ASR 和原始接近 (没破坏主说话人)
  4. 按分数排序，输出 top N 到演示包

输出:
  demo_for_boss/
    sample_{i}_{tier}_{name}/
      1_原始.wav
      2_分离_轨1.wav
      3_分离_轨2.wav
      info.txt
    README.txt
"""
import argparse
import glob
import os
import random
import re
import shutil
import sys
import warnings
import numpy as np
import onnxruntime as ort

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio, save_audio
from separate_baseline import separate_one, MODEL_PATH, SR, speaker_similarity
from device_utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VERIFY_MULTI_DIR = os.path.join(BASE_DIR, "verify_multi")
GREENPP_DIR = os.path.join(BASE_DIR, "recordings_cleaned_v2", "green++")
OUT_DIR = os.path.join(BASE_DIR, "demo_for_boss")


def normalize_text(s):
    return re.sub(r"[^\w一-鿿]", "", s)


def edit_distance(a, b):
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = cur
    return dp[m]


def char_diff(a, b):
    a, b = normalize_text(a), normalize_text(b)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return edit_distance(a, b) / max(len(a), len(b))


def score_separation(text_raw, text_t1, text_t2, sim):
    """
    分离质量评分。
    """
    chars_t1 = len(normalize_text(text_t1))
    chars_t2 = len(normalize_text(text_t2))
    score = 0.0
    reasons = []

    if chars_t1 >= 3 and chars_t2 >= 3:
        score += 3
        reasons.append("两路都有内容(+3)")
    elif chars_t1 >= 3 and chars_t2 == 0:
        reasons.append("仅轨1有内容")
    elif chars_t1 == 0 and chars_t2 >= 3:
        reasons.append("仅轨2有内容")
    else:
        reasons.append("两路都缺内容")

    if chars_t1 >= 2 and chars_t2 >= 2:
        diff = char_diff(text_t1, text_t2)
        if diff > 0.5:
            score += 2
            reasons.append(f"内容差异大(diff={diff:.2f},+2)")
        elif diff > 0.3:
            score += 1
            reasons.append(f"内容部分不同(diff={diff:.2f},+1)")

    if isinstance(sim, float) and not np.isnan(sim):
        if sim < 0.7:
            score += 2
            reasons.append(f"说话人余弦sim={sim:.2f}低(+2)")
        elif sim < 0.85:
            score += 1
            reasons.append(f"sim={sim:.2f}中等(+1)")

    if chars_t1 >= 3 and char_diff(text_raw, text_t1) < 0.3:
        score += 1
        reasons.append("主轨与原始相近(+1)")

    return score, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-success", type=int, default=3,
                        help="多说话人成功分离样本数")
    parser.add_argument("--n-control", type=int, default=2,
                        help="清晰单人对照样本数 (来自 green++)")
    parser.add_argument("--max-search", type=int, default=40,
                        help="最多扫描多少个 verify_multi 样本来找成功 case")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # 收集候选
    if not os.path.isdir(VERIFY_MULTI_DIR):
        print(f"找不到 {VERIFY_MULTI_DIR}, 先跑 is_multispeaker.py --save-multi-to verify_multi")
        sys.exit(1)
    multi_files = sorted(glob.glob(os.path.join(VERIFY_MULTI_DIR, "*.wav")))
    if not multi_files:
        print("verify_multi/ 里没文件")
        sys.exit(1)

    # 取前 max_search 个扫描
    if len(multi_files) > args.max_search:
        random.shuffle(multi_files)
        multi_files = multi_files[:args.max_search]

    # 加载模型
    print(f"扫描 {len(multi_files)} 个多说话人样本，找分离成功 case...")
    session = ort.InferenceSession(MODEL_PATH)
    device = get_device()
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
        disable_update=True, disable_log=True, disable_pbar=True,
        device=device,
    )

    def transcribe(samples):
        try:
            r = asr.generate(input=samples, fs=SR, batch_size_s=300)
            text = ""
            for item in r:
                if item.get("sentence_info"):
                    for sent in item["sentence_info"]:
                        text += sent.get("text", "")
                else:
                    text += item.get("text", "")
            return text.strip()
        except Exception:
            return ""

    # 扫描评分
    candidates = []
    for i, path in enumerate(multi_files, 1):
        try:
            raw, sr = load_audio(path, verbose=False)
            tracks = separate_one(session, raw)
            text_raw = transcribe(raw)
            text_t1 = transcribe(tracks[0])
            text_t2 = transcribe(tracks[1])
            sim = speaker_similarity(tracks)
            sim = sim if isinstance(sim, float) else 1.0

            score, reasons = score_separation(text_raw, text_t1, text_t2, sim)
            candidates.append({
                "path": path,
                "raw": raw,
                "tracks": tracks,
                "text_raw": text_raw,
                "text_t1": text_t1,
                "text_t2": text_t2,
                "sim": sim,
                "score": score,
                "reasons": reasons,
            })
            if i % 10 == 0:
                print(f"  扫描 {i}/{len(multi_files)}")
        except Exception as e:
            print(f"  跳过 {os.path.basename(path)}: {e}")

    # 排序，取 top N
    candidates.sort(key=lambda c: -c["score"])
    success_cases = candidates[:args.n_success]

    print(f"\n=== 找到的 top {len(success_cases)} 多说话人成功分离样本 ===")
    for c in success_cases:
        print(f"  score={c['score']:.0f}  sim={c['sim']:.2f}  {os.path.basename(c['path'])[:14]}")
        print(f"    原始: {c['text_raw']}")
        print(f"    轨1:  {c['text_t1']}")
        print(f"    轨2:  {c['text_t2']}")
        print(f"    评分: {','.join(c['reasons'])}")

    # 控制组（清晰单人）
    print(f"\n=== 加 {args.n_control} 个清晰单人对照样本 ===")
    if os.path.isdir(GREENPP_DIR):
        gp_files = sorted(glob.glob(os.path.join(GREENPP_DIR, "*.wav")))
        if gp_files:
            random.seed(args.seed)
            control_paths = random.sample(gp_files, min(args.n_control, len(gp_files)))
            control_cases = []
            for path in control_paths:
                raw, sr = load_audio(path, verbose=False)
                tracks = separate_one(session, raw)
                text_raw = transcribe(raw)
                text_t1 = transcribe(tracks[0])
                text_t2 = transcribe(tracks[1])
                control_cases.append({
                    "path": path, "raw": raw, "tracks": tracks,
                    "text_raw": text_raw, "text_t1": text_t1, "text_t2": text_t2,
                    "sim": float("nan"), "score": 0,
                })
                print(f"  {os.path.basename(path)[:14]}")
                print(f"    原始: {text_raw}")
                print(f"    轨1:  {text_t1}")
        else:
            control_cases = []
            print("  (green++ 还没生成，跳过)")
    else:
        control_cases = []
        print("  (green++ 还没生成，跳过)")

    # 写演示包
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR)

    summary_lines = [
        "=" * 80,
        "演示包说明 (混合多人成功分离 + 单人对照)",
        "=" * 80,
        "",
        f"包含 {len(success_cases) + len(control_cases)} 个样本:",
        f"  - 前 {len(success_cases)} 个: 多说话人场景，分离效果较好",
        f"  - 后 {len(control_cases)} 个: 清晰单人对照",
        "",
        "每个子文件夹里:",
        "  1_原始.wav        真实录音",
        "  2_分离_轨1.wav    分离模型输出主轨 (通常是按按钮的孩子)",
        "  3_分离_轨2.wav    分离模型输出副轨 (通常是背景声音)",
        "  info.txt          ASR 文本对比 + 评分细节",
        "",
        "听感建议:",
        "  1. 多人样本: 听 1_原始.wav，能听到几个孩子？分别在说什么？",
        "  2. 听 2_分离_轨1.wav，按按钮的那个孩子是否被独立保留？",
        "  3. 听 3_分离_轨2.wav，背景的另一个孩子是否被抠出来？",
        "  4. 单人对照: 验证分离不会破坏单人录音的内容",
        "",
    ]

    all_cases = success_cases + control_cases
    for idx, c in enumerate(all_cases, 1):
        path = c["path"]
        name = os.path.splitext(os.path.basename(path))[0]
        tag = "多人" if idx <= len(success_cases) else "单人对照"
        short = f"sample_{idx}_{tag}_{name[:10]}"
        item_dir = os.path.join(OUT_DIR, short)
        os.makedirs(item_dir, exist_ok=True)

        save_audio(os.path.join(item_dir, "1_原始.wav"), c["raw"], SR)
        save_audio(os.path.join(item_dir, "2_分离_轨1.wav"), c["tracks"][0], SR)
        save_audio(os.path.join(item_dir, "3_分离_轨2.wav"), c["tracks"][1], SR)

        info_lines = [
            f"样本 {idx}: {tag}",
            f"文件: {os.path.basename(path)}",
            f"时长: {len(c['raw'])/SR:.2f} s",
            "",
            "音频:",
            f"  1_原始.wav        所有人的真实录音",
            f"  2_分离_轨1.wav    主说话人分离结果",
            f"  3_分离_轨2.wav    背景说话人分离结果 (单人时为空)",
            "",
            "ASR 识别 (FunASR Paraformer):",
            f"  原始: {c['text_raw']}",
            f"  轨1:  {c['text_t1']}",
            f"  轨2:  {c['text_t2'] if c['text_t2'] else '(空)'}",
            "",
        ]
        if "reasons" in c:
            info_lines.append(f"分离质量评分: {c['score']:.0f}")
            info_lines.append(f"细节: {', '.join(c['reasons'])}")
            info_lines.append(f"两路说话人相似度: {c['sim']:.2f} (<0.7=不同人, >0.85=很像)")
        with open(os.path.join(item_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(info_lines))

        summary_lines.append("-" * 80)
        summary_lines.append(f"📁 {short}/")
        summary_lines.append(f"   原始: {c['text_raw']}")
        summary_lines.append(f"   轨1:  {c['text_t1']}")
        summary_lines.append(f"   轨2:  {c['text_t2'] if c['text_t2'] else '(空)'}")
        if "reasons" in c:
            summary_lines.append(f"   评分: {c['score']:.0f}  sim={c['sim']:.2f}")
        summary_lines.append("")

    with open(os.path.join(OUT_DIR, "README.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print(f"\n✓ 演示包: {OUT_DIR}/")
    print(f"   含 {len(all_cases)} 个样本 + README.txt")


if __name__ == "__main__":
    main()
