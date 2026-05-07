# ASR - 多人语音分离 + 中文 ASR

公共区域设备（一体机）问询场景下的语音处理项目。从嘈杂的多人语音中分离出问询人的声音，再做语音转文字。

## 目录结构

```
.
├── audiosep/                      # v1 主要代码（baseline 单脚本 demo）
│   ├── sampletest.py              # 原版分离脚本（直接调 ONNX 模型）
│   ├── separate_enhanced.py       # 增强版：RMS 归一 + 谱减降噪 + 预加重 + Wiener 软掩码
│   ├── evaluate.py                # 分离质量评估（混合一致性、能量分布、说话人区分度，可选 SI-SDR）
│   ├── asr_test.py                # 用 faster-whisper 转写各路音频
│   ├── asr_cer.py                 # ASR 字错率评估（带繁→简归一）
│   ├── resample_dataset.py        # ⭐ 数据重采样工具：MP4-in-WAV → 标准 WAV @ 16kHz（多进程 ffmpeg）
│   ├── 2in1.wav                   # 测试音频（2 人混合，~4 秒）
│   └── 分离_*.wav                 # 分离结果
├── ASR_pipeline_v2/               # v2 工程化升级版（参见该目录下 README）
└── model/                         # 模型权重（未入库，需自行放入）
    └── model.onnx
```

> **v2 工作请进入 `ASR_pipeline_v2/`**，包含：DFN3 降噪、FunASR/whisper 多引擎对比、多 Pipeline 横向评估等完整工具链。

## 环境

conda 环境 `ASR`，Python 3.10：

```bash
conda create -n ASR python=3.10 -y
conda activate ASR
pip install onnxruntime numpy soundfile resemblyzer librosa faster-whisper opencc-python-reimplemented
```

## 使用

```bash
# 原版分离
python audiosep/sampletest.py

# 增强版分离（推荐，效果显著更好）
python audiosep/separate_enhanced.py

# 评估分离质量
python audiosep/evaluate.py              # 原版
python audiosep/evaluate.py --enhanced   # 增强版
python audiosep/evaluate.py --ref s1.wav s2.wav   # 有参考时算 SI-SDR

# ASR 转写 + CER
python audiosep/asr_cer.py

# 数据重采样（dataset/ 中的 MP4-in-WAV → dataset_16k/ 16kHz 标准 WAV）
python audiosep/resample_dataset.py
```

## 当前优化效果

测试音频（2 人混合，4 秒）：

| 指标 | 原版分离 | 增强版分离 |
|---|---|---|
| 重建 SNR | 3.7 dB | 72.0 dB |
| 说话人余弦相似度 | 0.626 | 0.561 |
| 轨1 ASR CER (whisper-medium) | 50% | **6.2%** |

## 关键改进

不动模型，仅在输入预处理 + 输出后处理两端的优化（详见 `separate_enhanced.py`）：

1. **Wiener 软掩码** —— 把模型输出当作能量分布估计，乘回 mix 复频谱，保证 `s1+s2≈mix` 并抑制伪影
2. **RMS 能量归一** —— 让模型工作在熟悉的电平区间
3. **谱减降噪** —— 输入端去掉持续背景噪声
4. **预加重** —— 输入提亮高频，输出还原

## 下一步规划

当前是「2 路固定输出的多人盲分离（BSS）」，对真实业务的 N 人场景能力有限。后续转向**目标说话人提取（TSE）** pipeline：

```
原始音频 → VAD → Diarization → 选目标人时段 → ASR
```

候选技术栈：pyannote.audio 3.x、FunASR (Paraformer + CAM++)、SenseVoice。
