"""
FunASR pipeline 最小 demo
  VAD (fsmn-vad)  →  Diarization (cam++)  →  ASR (Paraformer-zh)  →  Punc (ct-punc)

特点：
  - 端到端中文优化，不分离波形，而是按时间戳标 speaker
  - 输出：每个 segment 的 [start, end, speaker_id, text]
  - 真实业务里，可在此基础上选「问询人」segments 拼起来送 ASR

首次运行会从 ModelScope 下载模型（约 1.5 GB）
"""
import os
import sys
import time
from audio_io import load_audio, save_audio

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def build_model():
    from funasr import AutoModel
    print("加载 FunASR 模型（首次会下载）...")
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",
        disable_update=True,
    )
    print("模型就绪。\n")
    return model


def transcribe(model, audio_path):
    """
    返回:
      [
        {"start": 0.0, "end": 1.2, "spk": 0, "text": "你好我想问..."},
        {"start": 1.0, "end": 1.5, "spk": 1, "text": "..."},
        ...
      ]
    """
    # 用 audio_io 统一加载，避免 FunASR 默认走 torchaudio/ffmpeg 在 Windows 上的兼容问题
    samples, sr = load_audio(audio_path, verbose=False)

    t0 = time.time()
    result = model.generate(
        input=samples,
        fs=sr,
        batch_size_s=300,
        return_spk_res=True,
    )
    elapsed = time.time() - t0

    segments = []
    for item in result:
        # FunASR 返回 sentence_info 时已带 spk
        for sent in item.get("sentence_info", []):
            segments.append({
                "start": sent.get("start", 0) / 1000.0,
                "end": sent.get("end", 0) / 1000.0,
                "spk": sent.get("spk", -1),
                "text": sent.get("text", "").strip(),
            })

    return segments, elapsed, result


def pretty_print(segments):
    if not segments:
        print("  (无识别结果)")
        return
    print(f"  {'time':<14}{'spk':<6}text")
    print("  " + "-" * 60)
    for s in segments:
        ts = f"{s['start']:5.2f}-{s['end']:5.2f}"
        print(f"  {ts:<14}spk{s['spk']:<3}{s['text']}")


def select_target_speaker(segments, strategy="longest"):
    """
    从 diarization 输出选「问询人」。
    策略：
      - "longest"   说话总时长最长的 speaker
      - "first"     第一个开始说话的 speaker
      - "loudest"   能量最大的 speaker（需要原音频，未在此实现）
    """
    if not segments:
        return None
    spk_durations = {}
    spk_first_time = {}
    for s in segments:
        spk = s["spk"]
        if spk < 0:
            continue
        dur = s["end"] - s["start"]
        spk_durations[spk] = spk_durations.get(spk, 0) + dur
        spk_first_time.setdefault(spk, s["start"])

    if not spk_durations:
        return None
    if strategy == "longest":
        return max(spk_durations, key=spk_durations.get)
    if strategy == "first":
        return min(spk_first_time, key=spk_first_time.get)
    raise ValueError(f"未知策略: {strategy}")


def extract_target_text(segments, target_spk):
    return "".join(s["text"] for s in segments if s["spk"] == target_spk).strip()


def main():
    # 默认评测：用 samples_test/ 第一个文件
    import glob
    samples = sorted(glob.glob(os.path.join(BASE_DIR, "samples_test", "*.m4a")) +
                     glob.glob(os.path.join(BASE_DIR, "samples_test", "*.wav")))
    default_audio = samples[0] if samples else os.path.join(BASE_DIR, "samples_test", "missing.wav")
    audio_path = sys.argv[1] if len(sys.argv) > 1 else default_audio
    audio_path = os.path.abspath(audio_path)
    print(f"输入音频: {audio_path}")

    # 校验/规范化
    samples, sr = load_audio(audio_path)
    print(f"长度 {len(samples)/sr:.2f}s, sr={sr}\n")

    # 构建并跑 pipeline
    model = build_model()
    segments, elapsed, _ = transcribe(model, audio_path)

    print(f"=== FunASR 分段结果（耗时 {elapsed:.1f}s）===")
    pretty_print(segments)

    print("\n=== 按说话人聚合 ===")
    spk_set = sorted({s["spk"] for s in segments if s["spk"] >= 0})
    for spk in spk_set:
        text = extract_target_text(segments, spk)
        dur = sum(s["end"] - s["start"] for s in segments if s["spk"] == spk)
        print(f"spk{spk}（共 {dur:.2f}s）: {text}")

    print("\n=== 目标说话人选择 ===")
    for strat in ("longest", "first"):
        tgt = select_target_speaker(segments, strat)
        if tgt is not None:
            text = extract_target_text(segments, tgt)
            print(f"  策略={strat}: spk{tgt} → {text}")


if __name__ == "__main__":
    main()
