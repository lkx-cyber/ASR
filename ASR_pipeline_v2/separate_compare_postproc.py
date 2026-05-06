"""
对 samples_test/ 的同一组音频，对比两套分离方案：
  A. 原版（裸 ONNX）：直接推理，无任何后处理
  B. 增强版：RMS归一 + 谱减 + 预加重 + Wiener 软掩码

输出对比表 + ASR 转写差异，定位增强后处理的真实增益与边际。
"""
import os
import sys
import glob
import time
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audio_io import load_audio, save_audio
from separate_baseline import (
    rms_normalize, preemphasis, deemphasis,
    spectral_denoise, wiener_refine,
    evaluate_separation, speaker_similarity,
    DATASET_DIR, MODEL_PATH, SR,
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_separated_compare")


def separate_raw(session, raw):
    """原版：直接送原音频到 ONNX，无后处理"""
    in_name = session.get_inputs()[0].name
    outs = session.run(None, {in_name: np.expand_dims(raw, 0)})
    return [o.squeeze().astype(np.float32) for o in outs]


def separate_enhanced(session, raw):
    """增强版：完整 pipeline"""
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.m4a")))
    if not files:
        print("未找到音频文件")
        return

    print("加载分离模型...")
    session = ort.InferenceSession(MODEL_PATH)

    print("加载 FunASR...")
    from funasr import AutoModel
    asr = AutoModel(
        model="paraformer-zh", vad_model="fsmn-vad",
        punc_model="ct-punc", spk_model="cam++",
        disable_update=True, disable_log=True, disable_pbar=True,
    )

    def transcribe(samples):
        result = asr.generate(input=samples, fs=SR, batch_size_s=300, return_spk_res=True)
        parts = []
        for item in result:
            if item.get("sentence_info"):
                for sent in item["sentence_info"]:
                    parts.append(sent.get("text", "").strip())
            else:
                parts.append(item.get("text", "").strip())
        return "".join(parts).strip()

    print()

    # 汇总用
    summary = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        short = name.replace("recording_", "")
        print("=" * 90)
        print(f"📁 {os.path.basename(path)}")

        mix, sr = load_audio(path, verbose=False)
        dur = len(mix) / sr
        print(f"   时长 {dur:.2f}s")

        # --- 原版 ---
        t0 = time.time()
        raw_sources = separate_raw(session, mix)
        raw_time = time.time() - t0
        raw_metrics = evaluate_separation(mix, raw_sources)
        raw_sim = speaker_similarity(raw_sources)
        raw_t1 = transcribe(raw_sources[0])
        raw_t2 = transcribe(raw_sources[1])

        # --- 增强版 ---
        t0 = time.time()
        enh_sources = separate_enhanced(session, mix)
        enh_time = time.time() - t0
        enh_metrics = evaluate_separation(mix, enh_sources)
        enh_sim = speaker_similarity(enh_sources)
        enh_t1 = transcribe(enh_sources[0])
        enh_t2 = transcribe(enh_sources[1])

        # 保存
        for i, s in enumerate(raw_sources):
            save_audio(os.path.join(OUT_DIR, f"{short}_原版_轨{i+1}.wav"), s, SR)
        for i, s in enumerate(enh_sources):
            save_audio(os.path.join(OUT_DIR, f"{short}_增强_轨{i+1}.wav"), s, SR)

        # 输出
        print(f"\n   {'指标':<14}{'原版':<32}{'增强版':<32}")
        print(f"   {'-'*78}")
        sim_r = f"{raw_sim:.3f}" if isinstance(raw_sim, float) else str(raw_sim)[:20]
        sim_e = f"{enh_sim:.3f}" if isinstance(enh_sim, float) else str(enh_sim)[:20]
        print(f"   {'重建SNR(dB)':<14}{raw_metrics['重建SNR(dB)']:<32.1f}{enh_metrics['重建SNR(dB)']:<32.1f}")
        print(f"   {'能量比':<14}{str(raw_metrics['能量比']):<32}{str(enh_metrics['能量比']):<32}")
        print(f"   {'说话人相似度':<12}{sim_r:<32}{sim_e:<32}")
        print(f"   {'耗时(s)':<14}{raw_time:<32.1f}{enh_time:<32.1f}")

        print(f"\n   ASR (FunASR):")
        print(f"   [原版-轨1]   {raw_t1}")
        print(f"   [原版-轨2]   {raw_t2}")
        print(f"   [增强-轨1]   {enh_t1}")
        print(f"   [增强-轨2]   {enh_t2}")
        print()

        summary.append({
            "file": short,
            "duration": dur,
            "raw_snr": raw_metrics["重建SNR(dB)"],
            "enh_snr": enh_metrics["重建SNR(dB)"],
            "raw_sim": raw_sim if isinstance(raw_sim, float) else -1,
            "enh_sim": enh_sim if isinstance(enh_sim, float) else -1,
            "raw_t1_len": len(raw_t1),
            "enh_t1_len": len(enh_t1),
            "raw_t2_len": len(raw_t2),
            "enh_t2_len": len(enh_t2),
        })

    # 总结表
    print("=" * 90)
    print("📊 汇总（看说话人相似度对比 + ASR 内容长度）")
    print(f"{'文件':<14} | {'原版相似度':<12} | {'增强相似度':<12} | "
          f"{'原版字数(轨1/轨2)':<18} | {'增强字数(轨1/轨2)':<18}")
    print("-" * 90)
    for s in summary:
        rs = f"{s['raw_sim']:.3f}" if s['raw_sim'] >= 0 else "n/a"
        es = f"{s['enh_sim']:.3f}" if s['enh_sim'] >= 0 else "n/a"
        print(f"{s['file']:<14} | {rs:<12} | {es:<12} | "
              f"{s['raw_t1_len']:>3}/{s['raw_t2_len']:<3} 字           | "
              f"{s['enh_t1_len']:>3}/{s['enh_t2_len']:<3} 字")


if __name__ == "__main__":
    main()
