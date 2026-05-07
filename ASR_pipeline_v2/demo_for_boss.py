"""
给领导的演示包

从 green++ (高质量人耳能听清楚) 抽 5 条，每条提供:
  - 1_原始.wav            清晰原录音 (人耳能听清)
  - 2_分离_轨1.wav         baseline ONNX 分离的主轨
  - 3_分离_轨2.wav         baseline ONNX 分离的副轨
  - info.txt              ASR 文本对比 + 简评

输出: demo_for_boss/
"""
import argparse
import glob
import os
import random
import sys
import warnings
import numpy as np
import onnxruntime as ort

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio, save_audio
from separate_baseline import separate_one, MODEL_PATH, SR
from device_utils import get_device

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "recordings_cleaned_v2", "green++")
SOURCE_DIR_FALLBACK = os.path.join(BASE_DIR, "recordings_cleaned", "green")
OUT_DIR = os.path.join(BASE_DIR, "demo_for_boss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", default=None,
                        help="抽样源 (默认 green++ 不存在则用 green)")
    args = parser.parse_args()

    src = args.source
    if src is None:
        if os.path.isdir(SOURCE_DIR):
            files_check = glob.glob(os.path.join(SOURCE_DIR, "*.wav"))
            if len(files_check) >= args.n:
                src = SOURCE_DIR
                print(f"📂 来源: green++ (已确认 {len(files_check)} 条 ≥ {args.n})")
        if src is None:
            src = SOURCE_DIR_FALLBACK
            print(f"📂 来源: green (green++ 还没跑够) {src}")

    files = sorted(glob.glob(os.path.join(src, "*.wav")))
    if not files:
        print(f"找不到 wav: {src}")
        sys.exit(1)

    random.seed(args.seed)
    sampled = random.sample(files, min(args.n, len(files)))
    print(f"抽样 {len(sampled)} 条\n")

    os.makedirs(OUT_DIR, exist_ok=True)

    # 加载模型
    print("加载分离模型 + FunASR...")
    sep_session = ort.InferenceSession(MODEL_PATH)

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
        except Exception as e:
            return f"<asr_err:{e}>"

    print("\n开始处理...\n")

    summary_lines = []
    summary_lines.append("=" * 80)
    summary_lines.append("演示包说明")
    summary_lines.append("=" * 80)
    summary_lines.append("")
    summary_lines.append("每个子文件夹是一条样本，包含 3 个 wav + 1 个 info.txt:")
    summary_lines.append("  1_原始.wav        — 真实录音 (人耳能听清的高质量样本)")
    summary_lines.append("  2_分离_轨1.wav    — 分离模型输出的第 1 路 (通常是主说话人)")
    summary_lines.append("  3_分离_轨2.wav    — 分离模型输出的第 2 路 (通常是背景或空)")
    summary_lines.append("  info.txt          — ASR 文字识别对比")
    summary_lines.append("")
    summary_lines.append("听感建议:")
    summary_lines.append("  1. 先听 1_原始.wav 知道孩子说了什么")
    summary_lines.append("  2. 再听 2_分离_轨1.wav 听分离后是否清晰、是否丢内容")
    summary_lines.append("  3. 听 3_分离_轨2.wav 看是否真把背景声音抠出来了")
    summary_lines.append("  4. 看 info.txt 比对 ASR 文本，判断分离对识别有没有帮助")
    summary_lines.append("")
    summary_lines.append("我们当前观察到的问题:")
    summary_lines.append("  - 分离对单人 PTT 数据没明显帮助 (识别字数基本持平)")
    summary_lines.append("  - 极端嘈杂 / 真重叠场景下分离效果有限")
    summary_lines.append("  - 高质量样本本身 ASR 直转就够准 (不需要分离)")
    summary_lines.append("")

    for idx, path in enumerate(sampled, 1):
        name = os.path.splitext(os.path.basename(path))[0]
        short = f"sample_{idx}_{name[:14]}"
        item_dir = os.path.join(OUT_DIR, short)
        os.makedirs(item_dir, exist_ok=True)

        # 加载
        raw, sr = load_audio(path, verbose=False)
        dur = len(raw) / sr

        # 分离
        try:
            tracks = separate_one(sep_session, raw)
        except Exception as e:
            print(f"[{idx}/{len(sampled)}] {name} 分离失败: {e}")
            continue

        # ASR
        text_raw = transcribe(raw)
        text_t1 = transcribe(tracks[0])
        text_t2 = transcribe(tracks[1])

        # 保存
        save_audio(os.path.join(item_dir, "1_原始.wav"), raw, sr)
        save_audio(os.path.join(item_dir, "2_分离_轨1.wav"), tracks[0], sr)
        save_audio(os.path.join(item_dir, "3_分离_轨2.wav"), tracks[1], sr)

        # 各路 RMS
        rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
        rms_raw = rms(raw)
        rms_t1, rms_t2 = rms(tracks[0]), rms(tracks[1])

        info_lines = [
            f"样本 {idx}: {name}",
            f"时长: {dur:.2f} s",
            "",
            "音频文件:",
            f"  1_原始.wav        RMS={rms_raw:.4f}",
            f"  2_分离_轨1.wav    RMS={rms_t1:.4f}",
            f"  3_分离_轨2.wav    RMS={rms_t2:.4f}",
            "",
            "ASR 识别 (用相同模型 FunASR Paraformer):",
            f"  原始: {text_raw}",
            f"  轨1:  {text_t1}",
            f"  轨2:  {text_t2}",
            "",
        ]
        # 对比观察
        if text_raw == text_t1 and not text_t2:
            info_lines.append("观察: 分离没改变内容，主说话人完整保留在轨1。")
        elif text_raw != text_t1 and text_t1:
            info_lines.append(f"观察: 分离后字符变化 (原 {len(text_raw)} 字 → 轨1 {len(text_t1)} 字)，可能引入错字或漏字。")
        elif not text_t1 and not text_t2:
            info_lines.append("观察: 分离后两路都没识别出内容，分离过程破坏了语音。")
        else:
            info_lines.append("观察: 见上文，请人耳听感判断。")

        with open(os.path.join(item_dir, "info.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(info_lines))

        # 累加到 summary
        summary_lines.append("-" * 80)
        summary_lines.append(f"📁 {short}/")
        summary_lines.append(f"   时长 {dur:.2f}s")
        summary_lines.append(f"   ASR(原始): {text_raw}")
        summary_lines.append(f"   ASR(轨1):  {text_t1}")
        summary_lines.append(f"   ASR(轨2):  {text_t2 if text_t2 else '(空)'}")
        summary_lines.append("")

        print(f"[{idx}/{len(sampled)}] {short}")
        print(f"  原始: {text_raw}")
        print(f"  轨1:  {text_t1}")
        print(f"  轨2:  {text_t2 if text_t2 else '(空)'}\n")

    # 写顶层 README
    readme_path = os.path.join(OUT_DIR, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    print("=" * 70)
    print(f"✓ 演示包已生成: {OUT_DIR}/")
    print(f"  - {len(sampled)} 个子文件夹 (每个含 3 个 wav + info.txt)")
    print(f"  - README.txt   (顶层说明 + 全部样本 ASR 对比)")
    print(f"\n📤 给领导发: 把整个 {os.path.basename(OUT_DIR)}/ 文件夹打包发过去即可")


if __name__ == "__main__":
    main()
