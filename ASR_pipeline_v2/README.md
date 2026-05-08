# ASR Pipeline v2

公共区域设备（一体机）问询场景下的中文语音处理项目。从嘈杂的多人语音中分离/提取出问询人的声音，再做语音转文字。

本项目位于 [MasterQiann/ASR](https://github.com/MasterQiann/ASR) 仓库的 `ASR_pipeline_v2/` 子目录下。同仓库根目录的 `audiosep/` 是 v1 单脚本 demo（baseline 参照）。

> 🆕 **新接手 / 跨机器开发请先读 [PROGRESS.md](PROGRESS.md)** —— 含当前状态、下一步、协作约定。

v2 是 v1 的工程化升级版，引入：

- 真实业务录音批处理
- FunASR / MossFormer2 / faster-whisper 多引擎集成
- 多信号目标说话人识别
- 数据集自动清洗（已对 3458 条真实录音清洗完毕）
- **DeepFilterNet3 前端降噪模块**（吞音敏感场景下的保守混合策略）
- **多 pipeline 横向对比框架**（baseline / 降噪 / FunASR / whisper 任意组合）
- **二代清洗 (clean_dataset_v2.py)**：基于 5 个新信号 (Silero VAD / 谱熵 / F0 / 双引擎一致性 / RMS 动态) 的更严格分桶
- **单/多人判别器 (is_multispeaker.py)**：滑窗 speaker embedding 相似度
- **路由 pipeline (pipeline_router.py)**：单人直转 / 多人分离选目标
- **目标说话人增强 (enhance_target.py)**：锚点+持续性掩码（实测对真重叠场景效果有限，文档已记录）
- **演示包生成 (find_separation_demo.py)**：从全量数据自动找有展示价值的多说话人分离案例

---

## 📁 项目结构

```
ASR_pipeline_v2/
├── README.md                            ← 本文件
├── PROGRESS.md                          ← 当前状态 + 下一步 + 跨机器协作约定
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
├── clean_dataset.py                     # v1 数据集清洗（13 个指标）
├── clean_dataset_v2.py                  # ⭐ v2 增强清洗（+5 个新信号: Silero VAD/谱熵/F0/双引擎/RMS动态）
├── is_multispeaker.py                   # 单/多人判别器（滑窗 speaker embedding 相似度）
│
│ -------- 路由 / 增强（实验性，已记录失败原因） --------
├── pipeline_router.py                   # 单/多人路由：单人 raw→FunASR / 多人 baseline→选目标→FunASR
├── test_pipeline_router.py              # 路由测试（实测改善有限，详见实验结论）
├── enhance_target.py                    # 目标人增强：锚点+持续性掩码（实测真重叠场景无效）
├── test_enhance_target.py               # 增强模块测试
│
│ -------- 演示包生成 --------
├── find_separation_demo.py              # ⭐ 自动从全量找"分离有展示价值"的多人 case
├── demo_for_boss.py                     # 抽样 5 条 (clean+/分离结果) 给老板的演示包
│
│ -------- 人工标注 --------
├── label_groundtruth.py                 # ⭐ Ground Truth 标注工具 (建评估金标准)
│
│ -------- 降噪（v2 新增） --------
├── denoise.py                           # DFN3 降噪模块（50% 干湿混合 + RMS 增益匹配，可复用）
│
│ -------- 分离评估 --------
├── test_separation_sample30.py          # 抽样 30 条评估分离效果（保存所有 wav 供人耳校验）
├── test_separation_all.py               # 全量评估，输出指标 CSV
├── device_utils.py                      # GPU/CPU 自动检测 + 设备管理
│
│ -------- 降噪 / FRCRN（v2 新增） --------
├── denoise.py                           # DFN3 降噪模块（已在上方列出）
├── frcrn_denoise.py                     # FRCRN（ClearerVoice，中文场景）封装
│
│ -------- 测试/评估脚本（v2 新增） --------
├── test_audio_methods_compare.py        # ⭐ 主入口：4 场景 × 8 方法横向对比
├── eval_with_groundtruth.py             # ⭐ 基于 GT 算 CER 的客观评估
├── test_multispeaker_accuracy.py        # 单/多人判别器准确率验证
├── test_denoise_sample.py               # 抽 10 条跑 DFN3，输出原始/降噪 wav 对比
├── test_denoise_strength.py             # 多种降噪强度 A/B（atten_lim / 干湿混合 / 增益匹配）
├── test_denoise_asr.py                  # 30 条 ASR 前后对比（whisper），统计字数/空率
├── measure_denoise_metrics.py           # 量化降噪客观指标（noise_floor / SNR / hf_noise / hnr 等）
├── test_funasr_vs_whisper.py            # whisper-medium vs FunASR Paraformer 对比（含降噪前后）
├── test_pipeline_compare.py             # 6 套 pipeline 横向对比（v1 老脚本，已被 test_audio_methods_compare 取代）
├── groundtruth.csv                      # 100 条人工标注样本（用于 eval_with_groundtruth）
│
│ -------- 数据 --------
├── samples_test/                        # ⭐ 团队自录的 4 段多说话人 m4a + 1 段 2in1.wav
│                                          16kHz 原生（非 8k 上采！），不存在空高频段问题
│                                          作为 S4 场景验证「未来硬件升级到 16k」之后效果
├── recordings_raw/                      # 真实业务 PTT 录音 (3458 条, 8kHz)
├── recordings_cleaned/                  # v1 清洗分桶 (13 信号)
│   ├── green/    优质 ( 60.2%, 2082 条)
│   ├── yellow/   可用 ( 27.4%,  949 条)
│   ├── orange/   边缘 (  4.8%,  167 条)
│   ├── red/      废弃 (  7.5%,  260 条)
│   └── report.csv / summary.txt
│
├── recordings_cleaned_v2/               # ⭐ v2 严格清洗分桶（推荐评估用）
│   ├── green++/  黄金集 ( 7.3%, 253 条)  ← 5/5 信号全过, 听感几乎全是高质量
│   ├── green+/   次高质 (52.8%, 1826 条) ← 3-4/5 过, 部分仍有杂质
│   ├── green/    一般   ( 0.1%, 3 条)
│   ├── yellow/   (27.4%,  949 条)
│   ├── orange/   (4.8%,  167 条)
│   ├── red/      (7.5%,  260 条)
│   ├── report_v2.csv      18 个指标 + 双引擎 ASR 文本 + 分桶
│   └── summary_v2.txt
│
├── verify_multi/                        # is_multispeaker 判定为多人的 61 条样本
└── multispeaker_test100.csv             # 单/多人判别器在 green 100 条上的报告
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

# 如果你有 NVIDIA GPU（强烈推荐，速度快 5-20x）：
pip uninstall torch torchaudio -y
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128  # RTX 50系
# 老一些的卡用 cu121 即可
```

GPU 检测会自动进行，无需改代码。设环境变量 `ASR_PIPELINE_DEVICE=cpu` 可强制 CPU。

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

### B. 在已清洗数据集上评估分离效果（核心场景）

我们提供两个专用脚本，都会自动从 `recordings_cleaned/green + yellow` 取样：

#### 🧪 脚本 1：抽样 30 条详查（推荐先跑）

```bash
python test_separation_sample30.py                 # 默认 30 条，含 baseline + MossFormer2
python test_separation_sample30.py --n 50          # 抽 50 条
python test_separation_sample30.py --no-mossformer # 仅 baseline
python test_separation_sample30.py --tiers green   # 只从 green 抽
python test_separation_sample30.py --seed 7        # 换种子得不同样本
```

**耗时（GPU 上）**：30 条 ~3-4 分钟，含 baseline + MossFormer2 + 5 次 ASR/条。

**输出**（`test_separation_sample30/`）：
```
test_separation_sample30/
├── sample_report.csv             每条的全部指标 + 自动分类
├── sample_summary.txt            统计 + 类别说明
└── {filename}/
    ├── mix.wav                   ← 原始录音
    ├── baseline_轨1.wav           ← ONNX + Wiener 后处理
    ├── baseline_轨2.wav
    ├── mossformer_轨1.wav         ← MossFormer2 SOTA
    └── mossformer_轨2.wav
```

**使用流程**：
1. 跑完后看 `sample_summary.txt` 中的"自动分类汇总"，判断 baseline / MF2 哪个对你的数据更适合
2. 进入子目录用 Audacity 或 VSCode Audio Preview 听 wav，**人耳校验自动分类是否准确**
3. 看 `sample_report.csv`，按 `baseline_class=success_multi` 筛真正的多人分离 case

#### 🧪 脚本 2：全量评估（决策依据）

```bash
# 默认: 仅 baseline, 无 ASR, 不存输出 wav (~30 分钟)
python test_separation_all.py

# 推荐: 失败 case 保存下来供听感校验（磁盘占用很小）
python test_separation_all.py --save-failures

# 加 ASR (~1.5 小时)
python test_separation_all.py --with-asr --save-failures

# 全套（baseline + MF2 + ASR）(~3-3.5 小时)
python test_separation_all.py --with-asr --with-mossformer --save-failures

# 调试: 只跑前 100 条
python test_separation_all.py --limit 100 --save-failures
```

**耗时表（GPU 上 3031 条 green+yellow）**：

| 命令 | 预估耗时 | 说明 |
|---|---|---|
| 默认 | ~30 分钟 | baseline + 相似度 + 能量比 |
| `--with-asr` | ~1.5 小时 | + 每条 3 次 ASR |
| `--with-mossformer` | ~1.5 小时 | + MF2 推理 |
| `--with-asr --with-mossformer` | **~3-3.5 小时** | 完整对比 |

⚠️ **这是离线批量评估时间，不是产线 UX**。产线每次只处理一个 PTT 请求，单条延迟约 1-2 秒，用户感知不到。

**输出**：
```
test_separation_all/
├── all_report.csv                  ← 全部录音的指标 (按 tier+sim 排序便于过滤)
├── all_summary.txt                 ← 分类统计 + 占比
└── failures/                       ← (--save-failures) 疑似失败的 case
    └── {filename}/                  让你听感复核分类是否准
        ├── mix.wav
        ├── baseline_轨1.wav
        └── baseline_轨2.wav
```

#### 📊 自动分类的 5 个类别（两个脚本通用）

| 类别 | 触发条件 | 含义 |
|---|---|---|
| `success_multi` | 相似度 < 0.7 + 两路都有 ASR 内容 | ✅ **真的成功分离了多人** |
| `single_clean` | 一路能量 < 5% + 另一路有内容 | ✅ 单人输入正确处理 |
| `failed_or_single` | 相似度 > 0.92 | ⚠️ 模型崩 / 输入本就单人 |
| `single_no_content` | 一路静音且另一路也无内容 | ⚠️ 录音质量差 |
| `ambiguous` | 中间状态 | 需人耳判断 |

**对 PTT 数据的预期分布**（业务大多数是单人）：
- `single_clean` 应是大头（baseline 上）
- `success_multi` 占比小但是**最有价值的 case**——这些是真正需要分离的多人录音
- `failed_or_single` 多了说明阈值要调或模型不够强

#### 🔍 重要发现：baseline > MossFormer2（对 PTT 数据）

实测 baseline 在你们的 PTT 数据上**普遍优于 MossFormer2**：

| 指标 | baseline (ONNX + Wiener) | MossFormer2 |
|---|---|---|
| 单人输入能量分布 | 99/1（一路接近静音） | 50/50（强行平分）|
| 听感 | 接近原 mix 音量 | 每路减半，细节丢失 |
| 多人输入分离 | 中等 | 更好（但多人 case 极少）|

**原因**：MossFormer2 是为「真实多人混合」训练的，单人输入是 OOD（out of distribution）；baseline 的 Wiener 软掩码会自动判定单人并把能量集中到一路。**95% 是单人的 PTT 业务场景下，baseline 更合适**。

---

### C. 在不同数据上测分离（修改 DATASET_DIR）

如果想跑非 cleaned 子集的数据，最简单方法是改脚本顶部 `DATASET_DIR`：

```python
# 在 separate_baseline.py 顶部改
DATASET_DIR = os.path.join(BASE_DIR, "recordings_cleaned", "green")
```

或者在命令行加 `--dataset` 参数（脚本未实现，可作为下一步迭代）。

### D. 跑数据清洗

**v1 清洗（13 个指标）**：
```bash
python clean_dataset.py                  # ~90 分钟 / 3458 条
python clean_dataset.py --limit 30       # 试跑
```

**v2 清洗（推荐）**——加 5 个新信号严格筛选：
```bash
python clean_dataset_v2.py               # ~120 分钟 / 3458 条 (GPU)
python clean_dataset_v2.py --limit 30    # 试跑
python clean_dataset_v2.py --skip-sensevoice  # 不跑双引擎一致性，更快
```

v2 新增的 5 个信号：
- **Silero VAD** 真实语音占比（筛掉误触录音）
- **谱熵 spectral entropy**（筛掉噪声主导段）
- **F0 voiced ratio**（验证真有人声）
- **双引擎 ASR 拼音 CER**（FunASR Paraformer vs SenseVoice 一致性）
- **RMS 动态范围**（筛掉平噪声）

输出 `recordings_cleaned_v2/` 含 6 个分桶（**新增 green++ / green+**）：
- `green++` 5/5 全过 → 听感校验确认"高质量"
- `green+` 3-4/5 过 → "参差不齐"，需进一步筛
- 阈值在脚本顶部，根据听感反馈可调

### E. 单/多人判别 + 演示包

### E. 单/多人判别 + 演示包

```bash
# 单文件单/多人判别
python is_multispeaker.py recordings_cleaned/green/abc.wav

# 批量, 把判为多人的样本复制到 verify_multi/
python is_multispeaker.py --dir recordings_cleaned/green --limit 100 --save-multi-to verify_multi/

# 给老板的 demo 包: 自动找 3 条多人成功分离 + 2 条单人对照
python find_separation_demo.py --n-success 3 --n-control 2

# 输出 demo_for_boss/  内含 5 条样本的 mix.wav + 分离轨 + ASR 对比
```

### F. Ground Truth 人工标注

**用途**：建评估金标准，让所有 pipeline 都能算客观 CER。

#### 启动

```bash
conda activate ASR
cd ASR_pipeline_v2

python label_groundtruth.py                    # 默认抽 100 条 green++
python label_groundtruth.py --n 50             # 改成 50 条
python label_groundtruth.py --source recordings_cleaned_v2/green+    # 换 tier
python label_groundtruth.py --output my_gt.csv                       # 改输出路径
python label_groundtruth.py --no-play          # 不自动播放, 用其他工具听
```

#### 操作流程

每条样本程序会自动播放音频 + 显示 ASR 自动识别文本作为参考。然后输入下面之一并回车：

| 操作 | 用途 |
|---|---|
| 直接输入正确文本 | 听到什么打什么 |
| `/a` | ASR 已经对了，一字不改采用 |
| `/r` | 重播当前音频 |
| `/s` | 跳过（不保存这条） |
| `/n 备注内容` | 给当前样本加备注（如 `/n 背景嘈杂`） |
| `/b` | 显示上一条信息（要改请手工编辑 CSV） |
| `/q` | 退出（已标的不会丢，下次自动续） |

#### 关键特性

- 每条立刻 flush 到 CSV，断电、Ctrl+C 都不会丢
- 重启自动跳过已标注的，从断点继续
- ASR 文本作为输入起点（多数 case 直接 `/a` 一键采用）

#### 输出

```
groundtruth.csv:
    filename, asr_text, correct_text, duration, labeled_at, notes
```

100 条预期标注时间约 2 小时（每条 60-90 秒）。

#### 标完之后

```python
# 用 groundtruth.csv 算任意 pipeline 的 CER
import csv
gt = {row["filename"]: row["correct_text"] for row in csv.DictReader(open("groundtruth.csv", encoding="utf-8-sig"))}

# 加载某个 pipeline 的输出, 对每条算 cer(pipeline_text, gt[filename])
# 平均 CER = 整体准确率
```

### G. ASR 引擎对比

```bash
# 单文件 FunASR demo
python asr_pipeline_funasr.py [audio_path]

# faster-whisper vs FunASR (需要 ../audiosep/ 即 v1 目录可用)
python asr_compare_engines.py
```

### H. 多信号目标说话人分析

```bash
# 必须先跑过 separate_baseline.py 生成分离结果
python analyze_target_signals.py
```

---

---

## 🔧 降噪与多 Pipeline 对比（v2 新增）

### 背景

我们的真实业务场景是**校园课间嘈杂环境** + **8kHz 采集** + **儿童语音为主**。前端降噪需要在「噪声压制」和「儿童弱辅音保护」之间取舍。本节工具就是为了：

1. 量化降噪的实际效果
2. 横向比较不同 pipeline 组合（降噪/分离/ASR 引擎）
3. 帮助决策最终生产配置

---

### G.1 降噪模块 `denoise.py`

封装好的 DeepFilterNet3 降噪模块，**经过 A/B 测试确定的最优配置**：

- 50% 干湿混合（避免吞音）
- RMS 增益匹配（保持原始响度）
- 防削波（peak ≤ 0.99）

```python
from denoise import Denoiser
dn = Denoiser()                  # 单例，首次约 0.1s
y = dn(x_16k)                    # x_16k: float32 numpy, mono, 16kHz
```

**特性**：
- 单例模式（避免重复加载模型）
- CPU 上 RTF ≈ 0.02（比实时快 50 倍）
- 自动 16k→48k→DFN3→16k 重采样

---

### G.2 降噪听感测试

#### 抽样测试（推荐先跑）

```bash
python test_denoise_sample.py                       # 默认 green/yellow 各 5 条
python test_denoise_sample.py --n-green 10 --n-yellow 10
```

**输出** `test_denoise/{tier}_{filename}/`：
- `1_原始.wav` — 原始录音
- `2_降噪.wav` — 降噪后

**控制台同时打印**每条样本的 RMS、噪声底变化和处理耗时。

⚠️ **过度抑制检测**：若降噪后 RMS 比原始低 30+ dB（如 -22 dB → -68 dB），说明 DFN3 把整段当成噪声压死了，需要对该样本退到更保守的设置。

#### 强度对比（针对吞音问题）

```bash
python test_denoise_strength.py --input recordings_cleaned/green/xxx.wav
```

对单条样本生成 5 种强度变体：
- `00_原始.wav`
- `06_干湿50%(原版).wav`
- `06a_干湿50%+RMS增益匹配.wav` ⭐ **推荐**
- `06b_干湿70%+RMS增益匹配.wav`
- `06c_干湿50%+峰值归一化_-3dBFS.wav`

用 Audacity / VSCode 对比挑最适合的。

---

### G.3 降噪客观指标量化 `measure_denoise_metrics.py`

```bash
# 先跑过 test_denoise_asr.py 生成样本对，然后：
python measure_denoise_metrics.py
```

**输出 8 个客观指标**（原始 vs 降噪）：

| 指标 | 方向 | 含义 |
|---|---|---|
| `noise_floor_db` | ↓ | 最低 10% 帧 RMS（背景噪声底） |
| `hf_noise_db_4-8k` | ↓ | 高频段能量（玩耍声/碰撞声主战场） |
| `lf_rumble_db_<150` | ↓ | 低频段能量（HVAC、风扇） |
| `spectral_flatness` | ↓ | 非语音段谱平坦度（越低越像有结构信号） |
| `speech_rms_db` | — | 语音段 RMS（应保持稳定） |
| `snr_db` | ↑ | 粗略信噪比 |
| `hnr_db` | ↑ | 谐波-噪声比（语音清晰度） |
| `crest_db` | ↑ | 波形动态范围 |

**实测结论**（cleaned 数据 30 条）：
- hf_noise: -4.3 dB（77% 样本改善）✅ 最强效果
- snr: +1.2 dB（70% 样本改善）
- noise_floor: -1.4 dB（70% 样本改善）
- 改善幅度小因为：① 50% 混合本身保守 ② 8kHz 数据已经较干净 ③ DFN3 训练数据偏成人

---

### G.4 ASR 引擎对比 `test_funasr_vs_whisper.py`

```bash
python test_funasr_vs_whisper.py                    # 默认 green/yellow 各 15
```

每条样本输出 4 个版本转写：whisper 原始/降噪 + FunASR 原始/降噪。

**实测结论（30 条样本）**：

| 引擎 | 平均字数 | 非空样本 | 幻觉数 |
|---|---|---|---|
| whisper 原始 | 7.4 | 23/30 | 多 |
| whisper 降噪 | 8.4 | 25/30 | 多 |
| **FunASR 原始** | **9.1** | **30/30** | **0** |
| **FunASR 降噪** | **9.4** | **30/30** | **0** |

**关键发现**：
- FunASR **召回率 100%**（whisper 7/30 输出空文本）
- FunASR **几乎不幻觉**（whisper 常见"謝謝觀看/我爱你/字幕志愿者"等模板）
- FunASR 速度比 whisper-medium 快 50-100 倍（RTF ~0.01）
- 降噪在 FunASR 上效果很小（+0.3 字 vs whisper +1.0 字）—— 因为 FunASR 本身抗噪

---

### G.5 多 Pipeline 横向对比 `test_pipeline_compare.py` ⭐

**最重要的对比工具**。同一份输入，6 套 pipeline 并跑：

| ID | Pipeline | 用途 |
|---|---|---|
| P0 | raw → whisper | 历史对照 |
| P1 | raw → FunASR | **最简最快** |
| P2 | DFN3 → FunASR | 加前端降噪 |
| P3 | baseline 分离 → FunASR | **现 baseline 路线** |
| P4 | DFN3 → baseline 分离 → FunASR | 降噪+分离 |
| P5 | baseline 分离 → whisper | baseline 早期形态 |

```bash
python test_pipeline_compare.py                     # 默认 green/yellow 各 10
python test_pipeline_compare.py --n-green 5 --n-yellow 5
python test_pipeline_compare.py --no-save-wavs      # 不存中间 wav，省磁盘
```

**输出**：
- 控制台：每条样本 6 个 pipeline 的转写对照
- `test_pipeline_compare/report.csv` — 全部数据 CSV
- `test_pipeline_compare/{tier}_{filename}/` — 每个 pipeline 的中间 wav，供人耳校验
  - `00_raw.wav`
  - `02_dfn3.wav`
  - `03_baseline_轨{1,2}.wav`
  - `04_dfn3_baseline_轨{1,2}.wav`

**典型结论（已跑过 15 条）**：
- 单人 PTT 场景下 **P1 (raw→FunASR)** 综合最好（最快、最准、无幻觉）
- 多人场景下 **P3 (baseline→FunASR)** 仍有价值（能分得开两人）
- 分离对单人 PTT **有副作用**：常引入"重复字"、"幻觉轨"
- DFN3 降噪在 FunASR pipeline 上**边际收益小**

**推荐生产架构**（按"单/多人判别 + 路由"）：
```
原始音频
  ↓ 单/多人判别
单人 → P1: raw → FunASR
多人 → P3: baseline 分离 → FunASR(各轨) → 选目标人
```

---

### G.6 ⭐ 音频方法横向对比 `test_audio_methods_compare.py`

**这是当前最重要的入口测试**——对比所有音频处理方法，覆盖 4 个真实业务场景。

#### 测试方法（4 场景 × 8 候选方法）

**场景：**

| 场景 ID | 数据来源 | 样本数 | 是否有 GT | 衡量指标 |
|---|---|---|---|---|
| **S1 单人 clean** | `groundtruth.csv` 中的 green++ 样本 | 5 | ✓ 有真值 | **CER**（字错率，越低越好）|
| **S2 多人** | `verify_multi/`（疑似多说话人）| 5 | ✗ | SNR + ASR 字数 + 说话人区分度 |
| **S3 嘈杂** | `recordings_cleaned_v2/yellow` 中 SNR 最低的 30 条随机抽 5 | 5 | ✗ | SNR 改善 + ASR 字数 |
| **S4 自录多人** | `samples_test/`（团队自己录的多人对话，**16kHz 原生**）| 5 | ✗ | SNR + ASR 字数 + 听感 |

**S4 是关键补充**：`samples_test/` 是**16kHz 原生采样**（非 8k 上采），不存在 4-8kHz 空高频段问题。在 S4 上跑，可以验证：
- 模型在「未来硬件升级到 16kHz」之后会怎么表现
- 区分"模型本身的能力"vs"被 8k 上采数据拖累的表现"
- 真实多人重叠场景下分离能力的天花板

**方法（按"是否做分离"分类）：**

| 方法 ID | Pipeline | 类别 |
|---|---|---|
| M0 | raw（不处理）| 基线 |
| M1 | DFN3 50%干湿混合 | 仅降噪 |
| M1b | FRCRN（中文场景）| 仅降噪 |
| M2 | 200ms 静音 padding | 仅前置 |
| M3 | DFN3 + pad200 | 降噪+前置 |
| M3b | **FRCRN + pad200** ⭐ | 降噪+前置 |
| M4 | baseline ONNX 分离 | 仅分离 |
| M5 | pad200 + baseline_sep | 前置+分离 |
| M6 | FRCRN + sep | 降噪+分离 |
| M7 | **FRCRN + pad200 + sep** ⭐ | 全配置 |

#### 怎么跑

```bash
# 默认: 4 场景各 5 条
python test_audio_methods_compare.py

# 指定 seed 复现实验
python test_audio_methods_compare.py --seed 42
```

#### 输出

```
test_audio_methods/<时间戳>/
├── summary.txt            场景×方法汇总（CER/SNR/字数 表格）
├── detail.csv             每条样本所有方法所有指标
└── wavs/<场景>/<filename>/
    ├── 00_原始.wav         基线
    ├── M0_raw_主.wav
    ├── M1_dfn3_主.wav
    ├── M1b_frcrn_主.wav
    ├── M3b_frcrn+pad200_主.wav
    ├── M5_pad+baseline_sep_主.wav   分离主轨
    ├── M5_pad+baseline_sep_副.wav   分离副轨
    ├── M7_frcrn+pad+sep_主.wav      ⭐ 当前最强候选
    └── gt.txt              S1 才有，真值文本
```

#### 决策框架

| 读数 | 判断 |
|---|---|
| S1 CER 低 + S2/S4 ASR 字数高 | **首选生产方案** |
| S1 CER 持平但 S3/S4 字数下降 | 处理过度，谨慎 |
| 仅 1-2 条样本拉开差距 | **不能做决策**，扩样本到 30 条/场景 |
| 多场景一致领先 | 真正的赢家 |

⚠️ **重要警告：5 条样本统计上不可靠**——单条幸运 case 就能让某方法看起来"领先"。生产决策前必须扩到 30 条/场景。

---

### G.7 ⭐ 基于 Ground Truth 的 CER 评估 `eval_with_groundtruth.py`

#### 用途

只针对**有 ground truth 的样本**算 CER（字错率），是**最严格的客观指标**。

#### 测试方法

输入：`groundtruth.csv`（100 条标注样本，来自 green++）
处理：每个候选 pipeline 在 GT 样本上跑 ASR，输出文本与真值做归一化后的字符级编辑距离对比
归一化策略：繁体→简体，全角→半角，去标点+空格，统一小写

#### 跑一次

```bash
# 跑全部已注册 pipeline
python eval_with_groundtruth.py

# 只跑指定 pipeline
python eval_with_groundtruth.py --pipelines P1_raw_funasr P3_pad200_funasr

# 调试用，限制前 N 条
python eval_with_groundtruth.py --limit 10
```

#### 输出

```
eval_results/<时间戳>/
├── summary.txt    总览（每个 pipeline 的整体 CER + CER 分布 + 对比）
└── detail.csv     每条样本的 ref/hyp/编辑距离/CER
```

#### 当前已知 baseline

| Pipeline | 整体 CER | 完美率 |
|---|---|---|
| P1 raw → FunASR | 10.70% | 43/98 (44%) |
| P2 DFN3 → FunASR | 10.40% | 40/98 (41%) |
| **P3 pad200 → FunASR** | **9.54%** | **48/98 (49%)** |
| P4 pad500 → FunASR | 9.79% | 46/98 |

注：FRCRN 和组合方案的 CER 还没正式跑这个脚本，要补。

---

### G.8 单/多人判别器准确率验证 `test_multispeaker_accuracy.py`

#### 用途

验证 `is_multispeaker.py` 是否能用——这决定了「单/多人路由」方案是否可行。

#### 测试方法

- **假设标签**：`groundtruth.csv` 中的样本默认为单人（来自 green++ 干净数据），`verify_multi/` 默认为多人（注：后者是判别器自己生成，是 circular 测试）
- 在两组样本上跑判别器
- 计算混淆矩阵 + 单人误判率（FP）
- 输出疑似 FP 列表供人耳复核

#### 跑一次

```bash
python test_multispeaker_accuracy.py
```

#### 当前已验证结论 ⚠️

```
GT 单人 min_sim 分布: 均值 0.55, 中位 0.54
verify_multi  min_sim 分布: 均值 0.55, 中位 0.56
（两个分布几乎完全重合！）

任何阈值下 FP 都 ≥ TP, 判别器无区分能力
```

**结论**：基于 resemblyzer 的当前判别器在儿童 + 8kHz 数据上失效，**路由方案不可行**。
→ 当前推荐"全员一刀切"使用 M3b 或 M7。

---

### G.9 FRCRN 降噪模块 `frcrn_denoise.py`

ClearerVoice-Studio 出品，中文场景训练，对儿童语音保留比 DFN3 好。

```python
from frcrn_denoise import FRCRNDenoiser
dn = FRCRNDenoiser()
y = dn(x_16k)   # 16kHz mono float32 → 同
```

实测 CER 改善：DFN3 在 cleaned 数据上 14.51%，**FRCRN 降到 8.89%**（5 样本，需扩验证）。

⚠️ 注意：FRCRN 在 4-8kHz 空高频段会填内容（HF 噪声 -112dB → -95dB），可能是"伪人声谐波"。听感校验后再决定是否上线。

---

### 🎯 评估流程推荐顺序

新接手项目时按以下顺序跑通：

```bash
# === 第一步：快速摸底 ===
python test_audio_methods_compare.py
# 5 分钟内看到 4 场景所有方法的对比表
# 输出: test_audio_methods/<时间戳>/summary.txt

# === 第二步：扩样本验证（生产决策前必做）===
# 改 test_audio_methods_compare.py 里 pick_samples 的样本数到 30
# 跑 30-60 分钟，得到统计稳定的对比

# === 第三步：CER 客观量化 ===
python eval_with_groundtruth.py
# 在 GT 100 条上对比所有 pipeline 的真实 CER

# === 第四步：听感校验 ===
# 进入 wavs/<scenario>/<filename>/ 听以下场景：
#   - S1 单人：FRCRN 是否吞字
#   - S3 嘈杂：分离是否漏听
#   - S4 自录多人：能否清晰分出两人
# 这一步不可替代——任何客观指标都会撒谎，唯有人耳不会

# === 第五步：边角验证 ===
python test_multispeaker_accuracy.py     # 判别器是否可用
python test_funasr_vs_whisper.py         # ASR 引擎是否切换
python measure_denoise_metrics.py        # 降噪客观指标
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

## 🧪 v2 阶段实验结论 (2026-05)

### 数据本质特征
- 真实业务录音是 **8kHz 上采样到 16kHz** 的 mp4-in-wav 格式
- 4-8kHz 频段几乎为空（重采样补零） → 影响所有依赖高频的模型
- 95% 是「主说话人 + 弱背景人声」的非对称混合，**不是均衡多人混合**
- "多说话人"场景里几乎全是**真重叠**（同帧两人同时发声），不是错峰

### 各方案验证结果

| 方案 | 状态 | 备注 |
|---|---|---|
| ❌ MossFormer2_SS_16K | **不可用** | 8kHz 上采数据上输出全是电流声/伪影。模型在 4-8kHz 空段凭空生成内容。低通修复无效。 |
| ❌ TSE 频域分离思路 | 同 MF2 风险 | 任何"生成型"模型在空高频段都会失败 |
| ⚠️ baseline ONNX + Wiener | **可用但有副作用** | 单人 PTT 场景下强行分两路，常常引入错字 (例: 三角洲→小猪)。多人 case 偶尔能成功分离。|
| ❌ enhance_target.py (锚点+持续性掩码) | **真重叠场景无效** | 窗口判断粒度太粗，目标人和干扰人在同帧时压不下去。30 条测试 ASR 字数变化 0%。 |
| ❌ pipeline_router.py (单/多人路由) | **改善有限** | 30 条对比 router 改对 1 / 改错 2 / 持平多。listen confirmed: baseline 分离对 PTT 数据多数情况是负贡献。 |
| ✅ FunASR Paraformer (raw 直转) | **当前最佳** | 30 条 100% 召回率, 0 幻觉, 平均 9.1 字/条. 不需要任何前处理. |
| ✅ DFN3 降噪 (50% 干湿混合) | **边际收益小** | 高频噪声 -4.3dB, 但 FunASR 本身抗噪强, 加 DFN3 ASR 字数 +0.3 |
| ✅ clean_dataset_v2 (5 信号严格清洗) | **核心工具** | 253 条 green++ 黄金集听感几乎全部高质量, 适合做 ground truth |

### 推荐生产架构

经过反复验证，**最简单的方案就是最好的**：

```
PTT 录音 → FunASR (raw) → 文本    ← 不要加任何前处理！
```

**不要做的事**:
- ❌ 加 MossFormer2 / 任何端到端生成型分离模型
- ❌ 加 baseline ONNX 分离（单人场景副作用 > 多人场景收益）
- ❌ 加复杂的目标说话人增强（实测无效）

**可以选做的事**:
- ⚠️ DFN3 前置降噪（收益小但无害，可作为可选项）
- ✅ 在数据清洗阶段用 v2 的 5 信号严格筛选高质量子集做评估

### 真正的"分离有展示价值"的 case
通过 `find_separation_demo.py` 在 verify_multi/ 上扫描，**约 5-10% 的多说话人录音能通过 baseline 分离得到两路有意义的不同内容**。这部分 case 在 `demo_for_boss/` 已演示。其他 90%+ 多说话人录音 baseline 分离没有展示价值（要么把 target 切坏、要么副轨是空气）。

### 下一步可探索方向
- [ ] 在 green++ 上抽 50 条建 ground truth，量化 raw FunASR 真实 CER
- [ ] 尝试将 4/5 通过的 1695 条 green+ 通过阈值微调升级到 green++（潜力大）
- [ ] 试 SenseVoice 替代 Paraformer（已加载，可对比）
- [ ] 专门为"主说话人 + 弱背景"场景训一个 8kHz 兼容的小模型（长期）

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
