"""
PTT pipeline 路由：单人直通 / 多人分离选目标

流程:
  1. is_multispeaker 判别
  2. 单人 → FunASR(raw) 直接转写
  3. 多人 → baseline 分离 → 用 RMS+最长段评分选目标轨 → FunASR 转写

输出:
  {
    "text": str,                  最终文本
    "branch": "single" | "multi",
    "is_multi_info": {...},       判别器返回
    "selected_track": int,         多人时选的是 0 还是 1
    "track_scores": [float, float] 多人时各轨得分
  }

用法:
    from pipeline_router import RouterPipeline
    pipeline = RouterPipeline()       # 加载所有模型
    result = pipeline.process(audio, sr)
    print(result["text"])

或直接调用单文件:
    python pipeline_router.py audio.wav
"""
import os
import sys
import warnings
import numpy as np
import onnxruntime as ort

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_io import load_audio
from is_multispeaker import classify as classify_multi, vad_energy
from separate_baseline import separate_one, MODEL_PATH, SR
from device_utils import get_device


def pick_target_track(tracks, w_rms=0.7, w_longest=0.3):
    """
    多人分离后从 N 路里选目标人。
    评分: 0.7*RMS_norm + 0.3*最长连续段_norm
    """
    n = len(tracks)
    # RMS
    rms = np.array([np.sqrt(np.mean(t ** 2)) for t in tracks])
    rms_norm = rms / (rms.max() + 1e-9)

    # 最长连续段
    longest = []
    for t in tracks:
        is_speech = vad_energy(t)
        max_run, cur = 0, 0
        for s in is_speech:
            if s:
                cur += 1
                max_run = max(max_run, cur)
            else:
                cur = 0
        longest.append(max_run)
    longest = np.array(longest, dtype=np.float32)
    longest_norm = longest / (longest.max() + 1e-9)

    scores = w_rms * rms_norm + w_longest * longest_norm
    return int(np.argmax(scores)), [float(s) for s in scores]


class RouterPipeline:
    def __init__(self, asr_model=None):
        self.device = get_device()
        print(f"[Router] device: {self.device}")

        # 加载 baseline 分离模型 (CPU)
        print("[Router] 加载 baseline ONNX...")
        self.sep_session = ort.InferenceSession(MODEL_PATH)

        # 加载 FunASR
        if asr_model is not None:
            self.asr = asr_model
        else:
            print(f"[Router] 加载 FunASR ({self.device})...")
            from funasr import AutoModel
            self.asr = AutoModel(
                model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc",
                disable_update=True, disable_log=True, disable_pbar=True,
                device=self.device,
            )

    def transcribe(self, audio):
        try:
            r = self.asr.generate(input=audio, fs=SR, batch_size_s=300)
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

    def process(self, audio, sr=SR):
        """
        Args:
            audio: float32 numpy mono 16k 或文件路径
            sr: 采样率
        Returns:
            dict
        """
        if isinstance(audio, str):
            audio, sr = load_audio(audio, verbose=False)

        # 1. 单/多人判别
        is_multi_info = classify_multi(audio, sr)

        # 2. 单人路径
        if not is_multi_info["is_multi"]:
            text = self.transcribe(audio)
            return {
                "text": text,
                "branch": "single",
                "is_multi_info": is_multi_info,
                "selected_track": None,
                "track_scores": None,
                "tracks": None,
            }

        # 3. 多人路径
        try:
            tracks = separate_one(self.sep_session, audio)
        except Exception as e:
            # 分离失败 → 退回 raw
            text = self.transcribe(audio)
            return {
                "text": text,
                "branch": "multi_separation_failed",
                "is_multi_info": is_multi_info,
                "selected_track": None,
                "track_scores": None,
                "tracks": None,
                "fallback_reason": str(e),
            }

        target_idx, scores = pick_target_track(tracks)
        text = self.transcribe(tracks[target_idx])

        return {
            "text": text,
            "branch": "multi",
            "is_multi_info": is_multi_info,
            "selected_track": target_idx,
            "track_scores": scores,
            "tracks": tracks,  # 调用方可保存做听感校验
        }


# CLI: 处理单个文件
def main():
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="音频文件路径")
    parser.add_argument("--save-tracks-to", help="多人时把分离轨保存到该目录")
    args = parser.parse_args()

    pipeline = RouterPipeline()
    result = pipeline.process(args.input)

    # 简化输出 (不打印 numpy)
    out = {k: v for k, v in result.items() if k != "tracks"}
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))

    if args.save_tracks_to and result.get("tracks"):
        from audio_io import save_audio
        os.makedirs(args.save_tracks_to, exist_ok=True)
        for i, t in enumerate(result["tracks"]):
            save_audio(os.path.join(args.save_tracks_to, f"track_{i}.wav"), t, SR)
        print(f"\n分离轨保存到: {args.save_tracks_to}/")


if __name__ == "__main__":
    main()
