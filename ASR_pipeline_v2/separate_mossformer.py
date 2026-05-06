"""
在 samples_test/ 上批量跑 MossFormer2_SS_16K，对比当前 ONNX 增强版的结果。

关键问题：MossFormer2 能不能救回当前模型崩盘的样本？

输出：
  - output_separated_mossformer/   MossFormer2 的所有分离结果
  - 控制台对比表：当前增强版 vs MossFormer2

依赖：先跑过 separate_baseline.py 才有 output_separated_enhanced/ 可对比
"""
import os
import sys
import glob
import time
import shutil
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio, save_audio
from separate_baseline import evaluate_separation, speaker_similarity, SR

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "samples_test")
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, "..", "model", "model.onnx"))
OUT_DIR = os.path.join(BASE_DIR, "output_separated_mossformer")
TEMP_DIR = os.path.join(BASE_DIR, "_tmp_mossformer_in")
EXISTING_DIR = os.path.join(BASE_DIR, "output_separated_enhanced")  # baseline 的结果


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.m4a")))
    if not files:
        print("未找到 m4a 文件")
        return

    print(f"找到 {len(files)} 个文件\n")

    # MossFormer2 不能直接读 m4a，先用 audio_io 转 16k 单声道 wav
    print("步骤1: 把 m4a 转换成 16k 单声道 wav...")
    wav_paths = []
    for m4a in files:
        name = os.path.splitext(os.path.basename(m4a))[0]
        wav_path = os.path.join(TEMP_DIR, f"{name}.wav")
        samples, sr = load_audio(m4a, verbose=False)
        save_audio(wav_path, samples, sr)
        wav_paths.append((name, wav_path, samples))
        print(f"  ✓ {name}.wav  ({len(samples)/sr:.2f}s)")
    print()

    # 加载 MossFormer2
    print("步骤2: 加载 MossFormer2_SS_16K...")
    from clearvoice import ClearVoice
    t0 = time.time()
    cv = ClearVoice(task="speech_separation", model_names=["MossFormer2_SS_16K"])
    print(f"  模型加载耗时 {time.time()-t0:.1f}s\n")

    # 跑分离
    print("步骤3: MossFormer2 分离...")
    for name, wav_path, _ in wav_paths:
        print(f"  处理 {name}.wav ...", end=" ", flush=True)
        t0 = time.time()
        cv(input_path=wav_path, online_write=True, output_path=OUT_DIR)
        print(f"耗时 {time.time()-t0:.1f}s")
    print()

    # ClearVoice 把结果写到 OUT_DIR/MossFormer2_SS_16K/{name}_s1.wav
    inner = os.path.join(OUT_DIR, "MossFormer2_SS_16K")

    # 加载 FunASR 跑 ASR
    print("步骤4: 加载 FunASR 跑 ASR...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", spk_model="cam++",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    def transcribe(samples):
        result = asr.generate(input=samples, fs=SR, batch_size_s=300, return_spk_res=True)
        text = ""
        for item in result:
            if item.get("sentence_info"):
                for sent in item["sentence_info"]:
                    text += sent.get("text", "")
            else:
                text += item.get("text", "")
        import re
        return text.strip(), len(re.sub(r"[^\w一-鿿]", "", text))

    print()

    # 对比
    print("=" * 100)
    print("📊 MossFormer2 vs 当前增强版 对比")
    print("=" * 100)

    summary = []

    for name, wav_path, raw in wav_paths:
        print(f"\n📁 {name}")
        print(f"   时长 {len(raw)/SR:.2f}s")

        # === MossFormer2 结果 ===
        m_s1_path = os.path.join(inner, f"{name}_s1.wav")
        m_s2_path = os.path.join(inner, f"{name}_s2.wav")
        if not (os.path.exists(m_s1_path) and os.path.exists(m_s2_path)):
            print(f"   ⚠️  MossFormer2 输出文件缺失，跳过")
            continue

        m_s1, _ = load_audio(m_s1_path, verbose=False)
        m_s2, _ = load_audio(m_s2_path, verbose=False)
        m_metrics = evaluate_separation(raw, [m_s1, m_s2])
        m_sim = speaker_similarity([m_s1, m_s2])
        m_t1, m_n1 = transcribe(m_s1)
        m_t2, m_n2 = transcribe(m_s2)

        # === 当前增强版结果 ===
        e_s1_path = os.path.join(EXISTING_DIR, f"{name}_轨1.wav")
        e_s2_path = os.path.join(EXISTING_DIR, f"{name}_轨2.wav")
        e_sim, e_metrics, e_t1, e_t2, e_n1, e_n2 = None, None, "", "", 0, 0
        if os.path.exists(e_s1_path) and os.path.exists(e_s2_path):
            e_s1, _ = load_audio(e_s1_path, verbose=False)
            e_s2, _ = load_audio(e_s2_path, verbose=False)
            e_metrics = evaluate_separation(raw, [e_s1, e_s2])
            e_sim = speaker_similarity([e_s1, e_s2])
            e_t1, e_n1 = transcribe(e_s1)
            e_t2, e_n2 = transcribe(e_s2)

        # 输出对比
        print(f"\n   {'指标':<14}{'当前增强版':<22}{'MossFormer2':<22}{'变化'}")
        print(f"   {'-'*70}")
        if e_metrics:
            de_snr = m_metrics["重建SNR(dB)"] - e_metrics["重建SNR(dB)"]
            print(f"   {'重建SNR(dB)':<14}{e_metrics['重建SNR(dB)']:<22.1f}{m_metrics['重建SNR(dB)']:<22.1f}{de_snr:+.1f}")
            print(f"   {'能量比':<14}{str(e_metrics['能量比']):<22}{str(m_metrics['能量比']):<22}")
            if isinstance(e_sim, float) and isinstance(m_sim, float):
                d_sim = m_sim - e_sim
                better = "✓" if d_sim < 0 else "✗"
                print(f"   {'说话人相似度':<12}{e_sim:<22.3f}{m_sim:<22.3f}{d_sim:+.3f} {better}")
            print(f"   {'ASR 字数(轨1)':<13}{e_n1:<22}{m_n1:<22}{m_n1 - e_n1:+d}")
            print(f"   {'ASR 字数(轨2)':<13}{e_n2:<22}{m_n2:<22}{m_n2 - e_n2:+d}")
        else:
            print(f"   (未找到当前增强版结果，仅显示 MossFormer2)")
            print(f"   重建SNR(dB):   {m_metrics['重建SNR(dB)']:.1f}")
            print(f"   说话人相似度:  {m_sim if isinstance(m_sim, str) else f'{m_sim:.3f}'}")
            print(f"   ASR 字数:      轨1={m_n1}  轨2={m_n2}")

        print(f"\n   ASR 转写对比:")
        if e_t1:
            print(f"   [增强-轨1]      {e_t1[:70]}")
            print(f"   [增强-轨2]      {e_t2[:70]}")
        print(f"   [Moss-轨1]      {m_t1[:70]}")
        print(f"   [Moss-轨2]      {m_t2[:70]}")

        summary.append({
            "name": name,
            "e_sim": e_sim, "m_sim": m_sim,
            "e_n1": e_n1, "e_n2": e_n2,
            "m_n1": m_n1, "m_n2": m_n2,
        })

    # 总览
    print("\n" + "=" * 100)
    print("📊 总览（说话人相似度越低越好；ASR 字数越多通常越好）")
    print(f"{'文件':<14}|{'增强版相似度':<14}|{'Moss相似度':<13}|{'增强版字数':<14}|{'Moss字数'}")
    print("-" * 100)
    for s in summary:
        es = f"{s['e_sim']:.3f}" if isinstance(s['e_sim'], float) else "n/a"
        ms = f"{s['m_sim']:.3f}" if isinstance(s['m_sim'], float) else "n/a"
        e_count = f"{s['e_n1']}/{s['e_n2']}"
        m_count = f"{s['m_n1']}/{s['m_n2']}"
        short = s['name'].replace("recording_", "")
        print(f"{short:<14}|{es:<14}|{ms:<13}|{e_count:<14}|{m_count}")

    # 清理临时
    print(f"\n清理临时目录 {TEMP_DIR}/")
    shutil.rmtree(TEMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
