# ASR Pipeline v2

公共区域设备（一体机）问询场景下的中文语音处理项目。从嘈杂的多人语音中分离/提取出问询人的声音，再做语音转文字。

本项目位于 [MasterQiann/ASR](https://github.com/MasterQiann/ASR) 仓库的 `ASR_pipeline_v2/` 子目录下。同仓库根目录的 `audiosep/` 是 v1 单脚本 demo（baseline 参照）。

v2 是 v1 的工程化升级版，引入：

- 真实业务录音批处理
- FunASR / MossFormer2 / faster-whisper 多引擎集成
- 多信号目标说话人识别
- 数据集自动清洗（已对 3458 条真实录音清洗完毕）

---

## 📁 项目结构

```
ASR_pipeline_v2/
├── README.md                            ← 本文件
├── .gitignore
│
│ -------- 通用工具 --------
├── audio_io.py                          # 音频读写：自动重采样、多声道、m4a/mp3/wav 兼容
│
│ -------- 分离 --------
├── separate_baseline.py                 # 当前 ONNX 模型 + 增强后处理 (Wiener 软掩码)
├── separate_mossformer.py               # MossFormer2_SS_16K (SOTA), 与 baseline 对比
├── separate_compare_postproc.py         # 原版裸推理 vs 增强版后处理 对比
│
│ -------- ASR --------
├── asr_pipeline_funasr.py               # FunASR 端到端: VAD + Paraformer + CAM++ + 标点
├── asr_compare_engines.py               # faster-whisper vs FunASR 对比 (依赖 v1 仓库)
│
│ -------- 分析与清洗 --------
├── analyze_target_signals.py            # 多信号目标人识别 (响度/时长/onset/声纹方差)
├── clean_dataset.py                     # 数据集自动清洗，分桶
│
│ -------- 数据 --------
├── samples_test/                        # 4 个 m4a 测试样本（10s 双说话人混合）
├── recordings_raw/                      # 真实业务 PTT 录音 (3458 条, 8kHz)
├── recordings_cleaned/                  # 清洗后分桶
│   ├── green/    优质 ( 60.2%, 2082 条)  ← 推荐评估用
│   ├── yellow/   可用 ( 27.4%,  949 条)
│   ├── orange/   边缘 (  4.8%,  167 条)
│   ├── red/      废弃 (  7.5%,  260 条)
│   ├── report.csv         全部 3458 条的指标 + 分桶 + 原因 + ASR 文本
│   └── summary.txt        总体统计
│
│ -------- 输出 (运行脚本生成) --------
├── output_separated_enhanced/           # baseline 分离结果
├── output_separated_mossformer/         # MossFormer2 分离结果
│
└── model_cache/                         # 模型权重缓存（gitignore, ~640MB）
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
conda create -n ASR python=3.10 -y
conda activate ASR

# 核心依赖
pip install onnxruntime numpy soundfile av soxr librosa
pip install resemblyzer webrtcvad-wheels                # 说话人嵌入 + VAD
pip install faster-whisper opencc-python-reimplemented  # whisper ASR + 繁简转换
pip install funasr modelscope torchaudio                # FunASR (Paraformer)
pip install clearvoice                                   # MossFormer2 分离
```

### 2. 模型权重

| 模型 | 来源 | 大小 | 自动下载？ |
|---|---|---|---|
| `model.onnx` (v1 BSS) | 自行提供，放 `../model/`（即仓库根 `model/`） | 220MB | ✗ |
| `MossFormer2_SS_16K` | ModelScope | 640MB | ✓ 首次跑 `separate_mossformer.py` |
| `faster-whisper medium` | HuggingFace | ~1.5GB | ✓ 首次跑相关脚本 |
| `Paraformer-zh / CAM++ / ct-punc` | ModelScope | ~1.5GB | ✓ 首次跑相关脚本 |

### 3. 一行命令跑通核心流程

```bash
# 测试样本上跑 baseline 分离 + ASR
python separate_baseline.py
```

输出 `output_separated_enhanced/` 中每个测试样本的分离 wav，控制台会打印重建 SNR、说话人相似度、能量比、ASR 转写。

---

## 📋 常见操作

### A. 在测试样本上跑分离

```bash
python separate_baseline.py            # 当前 ONNX + Wiener 后处理
python separate_mossformer.py          # MossFormer2 SOTA
python separate_compare_postproc.py    # 看后处理增益
```

### B. 在已清洗数据集上做评估

```bash
# 步骤 1: 选一个桶（推荐 green）的样本来测分离/ASR
ls recordings_cleaned/green/ | head    # 查看可用样本

# 步骤 2: 用 ASR 转写自动标注（FunASR 端到端）
python asr_pipeline_funasr.py recordings_cleaned/green/0007eee6bcd84b0fbeeb0e13b21f46aa.wav

# 步骤 3: 想批量测分离效果？把要测的文件复制成 samples_test/ 同名结构，
#         然后跑 separate_baseline.py，它会处理目录里所有 m4a / wav
```

如果要在 `recordings_cleaned/green/` 上批量评估分离，最简单的做法是临时把脚本里的 `DATASET_DIR` 指过去：

```python
# 在 separate_baseline.py 顶部改
DATASET_DIR = os.path.join(BASE_DIR, "recordings_cleaned", "green")
```

或者在命令行加 `--dataset` 参数（脚本未实现，可作为下一步迭代）。

### C. 跑数据清洗

```bash
# 完整清洗 recordings_raw/ (~90 分钟 / 3458 条)
python clean_dataset.py

# 试跑前 30 条
python clean_dataset.py --limit 30
```

输出在 `recordings_cleaned/`：
- 4 个桶子目录（green/yellow/orange/red），按照阈值自动分桶
- `report.csv`：每条的 13 项指标 + 分桶 + 原因 + ASR 文本
- `summary.txt`：总体统计 + Top 20 剔除原因

### D. ASR 引擎对比

```bash
# 单文件 FunASR demo
python asr_pipeline_funasr.py [audio_path]

# faster-whisper vs FunASR (需要 ../公司项目/audiosep/ 同级目录)
python asr_compare_engines.py
```

### E. 多信号目标说话人分析

```bash
# 必须先跑过 separate_baseline.py 生成分离结果
python analyze_target_signals.py
```

---

## 🎯 如何评估语音分离效果

**这是本项目的核心问题。我们采用「多维度评估」而非单一指标**。

### 维度 1：人耳听感（最重要、不可替代）★★★

**任何自动指标都会撒谎，唯有人耳不会。** 评估流程必须以听感为最终判据：

```bash
# 把分离结果直接用播放器打开
output_separated_enhanced/{filename}_轨{1,2}.wav
output_separated_mossformer/{filename}_s{1,2}.wav
```

听感问题清单：
- 两路是否真的是两个不同的人？（不是同一个人的两份噪声）
- 每路是否还能听到对方的声音？（串音）
- 是否有机械感、咕咕声、金属声等伪影？
- 声音内容是否完整？还是被切碎/吞字了？

### 维度 2：说话人余弦相似度（最可靠的自动指标）★★

```python
from resemblyzer import VoiceEncoder, preprocess_wav
encoder = VoiceEncoder()
emb1 = encoder.embed_utterance(preprocess_wav(track1, source_sr=16000))
emb2 = encoder.embed_utterance(preprocess_wav(track2, source_sr=16000))
similarity = emb1 @ emb2 / (norm(emb1) * norm(emb2))
```

| 相似度范围 | 含义 |
|---|---|
| < 0.50 | 优秀，两路是两个不同的人 |
| 0.50 - 0.70 | 良好，分得开但有少量串音 |
| 0.70 - 0.85 | 一般，明显串音 |
| > 0.85 | 较差，两路非常像 |
| > 0.92 | **分离失败**，两路本质是同一人或严重重叠 |

### 维度 3：能量分布 ★★

每路的 RMS 占比应反映真实场景：

| 占比 | 解读 |
|---|---|
| 50/50 | 两人音量相当（常见） |
| 70/30 | 一人较远或较小声（合理） |
| 90/10 或 100/0 | **某轨被压死**，分离失败 |

可能问题：Wiener 软掩码会把弱轨进一步压制（见实验结论）。

### 维度 4：ASR 转写一致性 ★★

不是看"字数多少"，而是看：

- **两路转写内容是否不同？** 同样的转写说明分得不彻底
- **每路转写是否语义连贯？** 碎片化输出说明分离质量差
- **跟原 mix 转写比，是否补全了被遮蔽的内容？** 分离的真正价值就在这里

```python
# 小贴士：FunASR 比 whisper 在中文口语和拟声词上更好
# whisper 在轨2 的"汪汪汪"上会输出"哇哇哇"或瞎编"我爱你 XD"
```

### 维度 5：重建 SNR（参考，不要单独看）★

```
重建 SNR = 10 * log10(mix² / (mix - α·(s1+s2))²)
```

**警告：这个指标具有欺骗性！**
- 如果只用 Wiener 软掩码，重建 SNR 必然 60+ dB（强制 s1+s2=mix），但分离质量可能很差
- 它只衡量"输出加起来等不等于 mix"，**不衡量"分得开不开"**

**只能作为辅助指标**，单独高不代表好。

### 维度 6：SI-SDR（仅在有 ground truth 时有效）

```python
# 需要参考音频（每个说话人的干净单轨）
# 实际业务里几乎不可能拿到，所以这个指标用不上
```

### 维度 7：分离失败检测（业务上线必备）

```python
def is_separation_failed(track1, track2, mix):
    """返回 True 表示这次分离不可用，应 fallback 到原 mix"""
    sim = speaker_similarity(track1, track2)
    if sim > 0.92:
        return True  # 分离崩盘
    
    energy_ratio = compute_energy_ratio(track1, track2)
    if min(energy_ratio) < 0.05:
        return True  # 某轨被压死
    
    return False
```

### 综合评分公式（推荐）

不同业务场景权重不同。**一体机问询场景**推荐：

```
quality_score = 0.4 × (1 - similarity)        # 相似度低越好
              + 0.3 × min(energy_ratio) × 2    # 能量平衡
              + 0.2 × (asr_chars > 5)          # ASR 有内容
              + 0.1 × subjective_listening      # 听感（手工标注）
```

---

## 📊 实验结论摘要

### 在 `samples_test/` (4 个测试样本) 上：

1. **当前 ONNX 模型 + 增强后处理在简单 case 上够用**，但在两人声纹相似度 >0.92 时崩盘（输出近乎空白）
2. **MossFormer2_SS_16K 能分离当前模型崩盘的 case**，但在简单 case 上不一定更好（无 Wiener 时能量分配差）
3. **多信号目标说话人识别（按 ASR 字数验证）**：
   - 响度 RMS、最长连续段：各 75% 命中率（排除分离失败 case 后均 100%）
   - 说话占比 / onset / 静音占比：50%
   - 声纹方差：25% （最差）
   - **推荐组合**：`0.7·RMS + 0.3·最长段`，简洁有效

### 在 `recordings_raw/` (3458 条真实录音) 上：

清洗结果：
- 🟢 **60.2%** 优质 (2082 条)
- 🟡 27.4% 可用 (949 条)
- 🟠 4.8% 边缘 (167 条)
- 🔴 7.5% 废弃 (260 条)

主要废弃原因：ASR 0 字 (95)、解码失败 (55)、ASR 1 字 (55)。

### 关键警告 ⚠️

1. **不要只看 ASR 字数**：whisper / Paraformer 会从噪声里编出合理文本（幻觉）。我们已用黑名单过滤常见幻觉模式，但仍要人耳复核。
2. **不要只看重建 SNR**：Wiener 软掩码必然让它飙到 60+ dB，但分离实际上可能很差。
3. **听感永远是最终标准**：抽样 30 条人工听是必做的校验步骤。

---

## 🛠️ 故障排查

### `model.onnx` 找不到
v1 BSS 模型权重需要自行提供并放到仓库根目录的 `model/model.onnx`（即 `ASR_pipeline_v2/../model/model.onnx`）。或修改 `separate_baseline.py` 顶部的 `MODEL_PATH`。

### `Couldn't find ffmpeg` 警告
不影响运行，audio_io 用 PyAV 解码 m4a/mp4，无需 ffmpeg。

### `webrtcvad` 安装失败
用 `pip install webrtcvad-wheels` 代替原版 `webrtcvad`（需要 C++ 编译）。

### `torchaudio` 加载警告
新 torchaudio 在 Windows 缺 torchcodec。我们的脚本绕开了这个：用 `audio_io.load_audio` 先加载成 numpy 再传给 FunASR。

### MossFormer2 下载慢
ModelScope 国内访问稳定，HuggingFace 可能需要代理。模型路径在 `model_cache/MossFormer2_SS_16K/`。

---

## 📝 路线图

下一步重点（按优先级）：

- [ ] 实装 PTT pipeline MVP：分离 → 多信号选目标人 → ASR
- [ ] 失败检测 + 兜底（相似度 >0.92 时 fallback 到原 mix）
- [ ] 双 ASR 引擎融合（whisper + FunASR）+ 后处理纠错
- [ ] 添加 SenseVoice ASR 引擎
- [ ] 抽样人工校验脚本（每桶随机抽 30 条）
- [ ] 业务热词偏置
- [ ] 给所有 separate_*.py 加 `--dataset` 命令行参数，避免修改源码

## 🤝 团队协作约定

- 每次跑实验先 `git pull`
- 修改阈值/参数前先在自己分支跑通，再合并
- 数据相关 PR 需附带听感校验结果（每桶至少 5 条）
- 不要 commit `model_cache/` 目录（已 gitignore）
- 推荐用 VSCode 的 Audio Preview 插件听 wav，效率高
