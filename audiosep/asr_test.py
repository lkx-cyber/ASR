"""
对原始混合音频、原版分离结果、增强版分离结果分别做语音转文字，对比 ASR 效果。

用 faster-whisper（CPU），中文模型默认 medium。
首次运行会从 HuggingFace 下载模型权重（约 1.5 GB）。
"""
import os
import glob
import time
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = "small"      # tiny / base / small / medium / large-v3，按需切换
LANGUAGE = "zh"

FILES = [
    ("混合音频", os.path.join(BASE_DIR, "2in1.wav")),
] + [
    (f"原版-轨{i+1}", p) for i, p in enumerate(sorted(glob.glob(os.path.join(BASE_DIR, "分离_说话人_*.wav"))))
] + [
    (f"增强版-轨{i+1}", p) for i, p in enumerate(sorted(glob.glob(os.path.join(BASE_DIR, "分离_增强_说话人_*.wav"))))
]


def transcribe(model, path):
    segments, info = model.transcribe(
        path,
        language=LANGUAGE,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    text = "".join(seg.text for seg in segments).strip()
    return text, info.duration


def main():
    print(f"加载 faster-whisper 模型 [{MODEL_SIZE}]（首次会下载）...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("模型就绪。\n")

    print("=" * 60)
    for label, path in FILES:
        if not os.path.exists(path):
            continue
        t0 = time.time()
        text, dur = transcribe(model, path)
        elapsed = time.time() - t0
        print(f"[{label}] {os.path.basename(path)} ({dur:.1f}s, ASR耗时 {elapsed:.1f}s)")
        print(f"  -> {text if text else '(无识别结果)'}")
        print()


if __name__ == "__main__":
    main()
