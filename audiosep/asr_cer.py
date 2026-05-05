"""带 CER 计算的 ASR 测试，与人工提供的参考文本对比"""
import os
import glob
import time
from faster_whisper import WhisperModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SIZE = "medium"
LANGUAGE = "zh"

# Ground truth
REF1 = "我考试考得太差了我都有点不想活了"
REF2 = "我养了一只小狗汪汪汪汪汪"

# (label, path, [可能的参考列表 —— 因为分离后两路顺序不固定，取较小 CER])
TASKS = [
    ("混合音频", os.path.join(BASE_DIR, "2in1.wav"), [REF1, REF2]),
    ("原版-轨1", os.path.join(BASE_DIR, "分离_说话人_1.wav"), [REF1, REF2]),
    ("原版-轨2", os.path.join(BASE_DIR, "分离_说话人_2.wav"), [REF1, REF2]),
    ("增强版-轨1", os.path.join(BASE_DIR, "分离_增强_说话人_1.wav"), [REF1, REF2]),
    ("增强版-轨2", os.path.join(BASE_DIR, "分离_增强_说话人_2.wav"), [REF1, REF2]),
]


_t2s = None
def to_simplified(s):
    global _t2s
    if _t2s is None:
        from opencc import OpenCC
        _t2s = OpenCC("t2s")
    return _t2s.convert(s)


def normalize(s):
    """繁→简，去掉标点和空格，只保留中文/字母/数字"""
    import re
    s = to_simplified(s)
    return re.sub(r"[^\w一-鿿]", "", s)


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m]


def cer(hyp, ref):
    h, r = normalize(hyp), normalize(ref)
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return edit_distance(h, r) / len(r)


def transcribe(model, path):
    segments, info = model.transcribe(
        path, language=LANGUAGE, beam_size=5, vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )
    return "".join(seg.text for seg in segments).strip()


def main():
    print(f"加载 faster-whisper [{MODEL_SIZE}]...")
    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    print("就绪。\n")

    print(f"参考文本1（说话人1）: {REF1}")
    print(f"参考文本2（说话人2）: {REF2}")
    print("=" * 70)

    rows = []
    for label, path, refs in TASKS:
        if not os.path.exists(path):
            continue
        text = transcribe(model, path)
        cers = [(cer(text, r), r) for r in refs]
        best_cer, best_ref = min(cers, key=lambda x: x[0])
        rows.append((label, text, best_ref, best_cer))

    print(f"{'来源':<14}{'最佳CER':>10}  转写结果")
    print("-" * 70)
    for label, text, ref, c in rows:
        print(f"{label:<14}{c*100:>8.1f}%  {text}")

    print("\n说明：CER = 字符错误率（编辑距离/参考字数），越低越好")


if __name__ == "__main__":
    main()
