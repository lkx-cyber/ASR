"""
统一音频读写工具。
所有输入音频都会被规范化为：单声道、16kHz、float32 numpy 数组。

支持：
  - 任意采样率自动重采样
  - 立体声/多声道转单声道（默认平均，可选取通道）
  - 多种格式（wav / flac / ogg / m4a 等 soundfile 支持的）
  - mp3 走 librosa（更稳）
"""
import os
import numpy as np
import soundfile as sf

DEFAULT_SR = 16000


def load_audio(path, target_sr=DEFAULT_SR, channel="mean", verbose=True):
    """
    Args:
      path: 输入文件路径
      target_sr: 目标采样率（默认 16k，分离/ASR 通用）
      channel: 多声道策略
        - "mean"        默认，所有通道求平均
        - 0/1/...       取指定通道
        - "max_energy"  取能量最大的通道（适合多麦阵列里有一只麦最靠近说话人）
      verbose: 是否打印转换日志

    Returns:
      (samples: np.ndarray[float32, mono], sr: int)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    ext = os.path.splitext(path)[1].lower()
    if ext in (".m4a", ".mp4", ".aac"):
        data, sr = _load_with_pyav(path)
    elif ext == ".mp3":
        try:
            data, sr = _load_with_pyav(path)
        except Exception:
            data, sr = _load_with_librosa(path)
    else:
        try:
            data, sr = sf.read(path, always_2d=False)
        except Exception:
            try:
                data, sr = _load_with_pyav(path)
            except Exception:
                data, sr = _load_with_librosa(path)

    if data.dtype != np.float32:
        data = data.astype(np.float32)

    # ---- 多声道处理 ----
    if data.ndim == 2:
        n_ch = data.shape[1]
        if channel == "mean":
            data = data.mean(axis=1)
            if verbose:
                print(f"[audio_io] {n_ch}ch → mono (mean)")
        elif channel == "max_energy":
            energies = np.mean(data ** 2, axis=0)
            idx = int(np.argmax(energies))
            data = data[:, idx]
            if verbose:
                print(f"[audio_io] {n_ch}ch → mono (channel {idx}, energy max)")
        elif isinstance(channel, int):
            if channel >= n_ch:
                raise ValueError(f"channel {channel} out of range for {n_ch}-channel audio")
            data = data[:, channel]
            if verbose:
                print(f"[audio_io] {n_ch}ch → mono (channel {channel})")
        else:
            raise ValueError(f"未知 channel 策略: {channel}")

    # ---- 重采样 ----
    if sr != target_sr:
        data = _resample(data, sr, target_sr)
        if verbose:
            print(f"[audio_io] {sr} Hz → {target_sr} Hz")
        sr = target_sr

    # ---- 防止极端幅值 ----
    peak = np.max(np.abs(data))
    if peak > 1.0:
        if verbose:
            print(f"[audio_io] peak={peak:.3f} > 1.0，归一到 0.99")
        data = data * (0.99 / peak)

    return data.astype(np.float32), sr


def save_audio(path, samples, sr=DEFAULT_SR, subtype="PCM_16"):
    """保存为 wav。默认 16-bit PCM，便于其他工具读取。"""
    samples = np.asarray(samples, dtype=np.float32)
    peak = np.max(np.abs(samples))
    if peak > 1.0:
        samples = samples * (0.99 / peak)
    sf.write(path, samples, sr, subtype=subtype)


def _resample(x, sr_from, sr_to):
    """优先用 soxr（高质量+快），回退到 librosa。"""
    try:
        import soxr
        return soxr.resample(x, sr_from, sr_to, quality="HQ").astype(np.float32)
    except ImportError:
        import librosa
        return librosa.resample(x, orig_sr=sr_from, target_sr=sr_to).astype(np.float32)


def _load_with_librosa(path):
    import librosa
    data, sr = librosa.load(path, sr=None, mono=False)
    if data.ndim == 2:
        data = data.T  # librosa 是 (n_ch, T)，统一成 (T, n_ch)
    return data.astype(np.float32), sr


def _load_with_pyav(path):
    """用 PyAV (libav 绑定) 解码 m4a / mp4 / aac / 等容器。"""
    import av
    container = av.open(path)
    audio_stream = next(s for s in container.streams if s.type == "audio")
    sr = audio_stream.rate
    n_ch = audio_stream.channels

    chunks = []
    for frame in container.decode(audio_stream):
        arr = frame.to_ndarray()  # 形状可能是 (channels, samples) 或 (samples,)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        chunks.append(arr)
    container.close()

    if not chunks:
        raise RuntimeError(f"PyAV 未解码出任何音频帧: {path}")

    data = np.concatenate(chunks, axis=1)  # (channels, total_samples)
    # 转 float32 [-1, 1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    # 统一成 (T,) 单声道 或 (T, n_ch) 多声道
    if n_ch == 1:
        data = data.reshape(-1)
    else:
        data = data.T  # (T, n_ch)
    return data, sr


def info(path):
    """快速查看音频元信息（不读全部数据）。"""
    try:
        i = sf.info(path)
        return {
            "frames": i.frames,
            "samplerate": i.samplerate,
            "channels": i.channels,
            "format": i.format,
            "subtype": i.subtype,
            "duration_sec": i.frames / i.samplerate,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python audio_io.py <音频路径>")
        sys.exit(0)
    p = sys.argv[1]
    print("元信息:", info(p))
    x, sr = load_audio(p)
    print(f"已加载: shape={x.shape}, sr={sr}, dtype={x.dtype}, RMS={np.sqrt(np.mean(x**2)):.4f}")
