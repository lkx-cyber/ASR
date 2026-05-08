# 当前进度

> **作用**：跨机器/跨人协作的"留言板"。每次工作结束前更新，下次开始前先读。
>
> **更新规则**：每个 git push 之前最多花 1 分钟同步本文件。

---

## 📅 最近一次工作

- **时间**：2026-05-08 下午
- **机器**：MacBook (CPU)
- **完成**：
  - 写 `eval_with_groundtruth.py`，基于 100 条 GT 算 CER（含文本归一化）
  - 测了 FunASR raw 的真实 baseline CER：**10.70%**（之前算的 5.47% 是错的）
  - 验证 200ms 静音 padding 有效：CER 10.70% → **9.54%**
  - 接入 FRCRN（ClearerVoice 中文降噪），实测 CER 14.51% → 8.89%（5 样本）
  - 写 `test_audio_methods_compare.py` 主入口对比框架（4 场景 × 8 方法）
    - S1 单人 clean / S2 多人 / S3 嘈杂 / S4 自录多人(samples_test 16k 原生)
  - 写 `test_multispeaker_accuracy.py`，验证判别器
  - 修 `label_groundtruth.py` 的 duration bug（MP4-in-WAV 兼容）

---

## 🎯 当前状态

### 数据
- `recordings_raw/`: 3458 条原始 PTT 录音 (8kHz)
- `recordings_cleaned/`: v1 清洗
- `recordings_cleaned_v2/`: v2 严格清洗
  - **green++ 253 条** ⭐ 黄金集
  - green+ 1826 条
  - yellow/orange/red: 不达标
- `verify_multi/`: 61 条多说话人样本
- `groundtruth.csv`: 100 条人工标注（98 有效, 2 跳过）⭐

### 工具链
- `clean_dataset_v2.py` — 5 信号严格清洗
- `label_groundtruth.py` — 人工标注（duration bug 已修）
- **`test_audio_methods_compare.py`** ⭐ 主入口：4 场景 × 8 方法横向对比
- **`eval_with_groundtruth.py`** ⭐ 基于 GT 算 CER
- `test_multispeaker_accuracy.py` — 判别器准确率
- `denoise.py` — DFN3 降噪
- **`frcrn_denoise.py`** — FRCRN 降噪（中文场景）

### 已验证的结论

#### 早期（2026-05-07）
- ❌ MossFormer2 在 8kHz 上采数据电流声，不可用
- ❌ baseline 分离对单人 PTT 有副作用
- ❌ enhance_target 对真重叠场景无效
- ❌ pipeline_router 改善有限
- ✅ FunASR raw 直转可作为 baseline
- ⚠️ DFN3 降噪边际收益小

#### 最新（2026-05-08）⭐
- **FunASR raw + 200ms padding 真实 CER 9.54%**（GT 上现场重测）
- ⭐ **FRCRN 显著优于 DFN3**（CER 14.51% → 8.89%，5 样本）
  - FRCRN 副作用：4-8kHz 空高频段会填内容（HF -112dB → -95dB），可能是伪谐波
- ⭐ **M7 (FRCRN+pad200+sep) 是当前候选最优**（仅 5 样本，待扩验证）
  - S1 单人 CER 2.22% / S2 多人 SNR 65.7dB / S3 嘈杂 SNR 70.9dB
  - 但 S3 嘈杂 ASR 字数比 M5 少，需听感校验
- ❌ **is_multispeaker 在儿童 8k 数据上完全失效**
  - 单人/多人 min_sim 分布完全重合（均值都 0.55）
  - 任何阈值都无法区分（FP=TP）
  - **路由方案搁置**，回归全员一刀切

---

## 🔜 下一步要做的（按优先级）

### 🔥 高优先级 — Mac/任意机器（CPU 即可，**收尾必做**）

- [ ] **削波检测 + 修复（declipping）**
  - 业务背景：用户最早反馈"麦克风炸、波形顶部被削平、听起来炸裂沙沙刺耳"
  - 是音频处理侧最早提出但**至今没做**的需求，不做的话交差有缺口
  - 实现思路：
    - 检测：扫描连续 `|x| > 0.99` 的采样段，N≥3 个采样点视为削波
    - 修复：三次样条插值，从两侧"未削波"采样点拟合
    - 代码 < 100 行，纯 numpy/scipy，CPU 实时
  - 预期：受影响样本 CER -3~5%，整体 -0.3~0.5%
  - 工作量：1-2 天
  - 验证：用 test_audio_methods_compare.py 加 M8_declip 方法对比

- [ ] **VAD 智能切片（替代/增强 200ms 静态 padding）**
  - 当前 200ms padding 是静态的——不论录音边界质量如何都加固定空白
  - 智能切片：用 VAD（如 Silero）识别真实语音段，自动切除前后噪声/静默/喘气
  - 比静态 padding 更精细，能同时处理"开头吞字"和"结尾尾音"问题
  - 实现：Silero VAD（已有依赖）→ 找首尾语音段 → 前后各加 200ms 缓冲 → 中间无效段保留
  - 预期：CER -0.3%，且会减少 ASR 在边缘段瞎猜
  - 工作量：1 天
  - 验证：作为 M9_vad_trim 加入对比框架

### 🔥 高优先级 — 等 Windows GPU 跑

- [ ] **N=30/场景 大样本对比验证**（5 样本结论统计上不可靠）
  ```bash
  cd ASR_pipeline_v2
  python test_audio_methods_compare.py --n 30
  ```
  - 重点验证：M7 在 N=30 下是否仍领先 M5/M3b
  - 如果 M7 持续领先 → 替换 denoise.py 默认从 DFN3 改 FRCRN，更新生产推荐
  - 如果 M7 退化 → 守 M5 或 M3b，不上 FRCRN

- [ ] **eval_with_groundtruth.py 加 FRCRN 和 M7 pipeline**
  ```bash
  python eval_with_groundtruth.py
  ```
  - 当前已注册 P1/P2/P3/P4，未注册 FRCRN/M7
  - 加进去能拿"GT CER"客观数字（不是 SNR/字数）

### 中优先级

- [ ] **听感校验 FRCRN 输出**
  - 进 `test_audio_methods/<时间戳>/wavs/S1_单人clean/<filename>/` 听
  - 重点：M1b_frcrn_主.wav vs M0_raw 听是否有"伪高频"
  - 决定 FRCRN 是否能上线

- [ ] **扩展 green++ 到 1700+**（探索 green+ 升级）
  - 抽 50 条 4/5 通过样本人工听
  - 80% 高质量则放宽 v2 阈值

- [ ] **写给领导的项目总结文档**
  - 整理音频处理侧的完整发现 + 决策 + 限制
  - 推荐生产配置 + 后续路线（硬件 / ASR / 业务侧）
  - 结论：分离/降噪模型微调在当前数据条件下不可行

### 长期 / 不在本组职责范围

- [ ] 等业务侧硬件升级到 16kHz 真采（领导提过有可能）
- [ ] 8kHz 原生方案？（暂不做，按补到 16k 路线走）
- [ ] ASR 微调 / hotword 偏置（属于 ASR 团队，非音频组）
- [ ] 关键词 / 紧急情绪检测分支（业务侧需求，单独项目）
- [ ] 声纹身份识别（业务侧需求）

---

## ❓ 待决策 / 阻塞

- **5 样本结论不够稳**——所有"M4/M7 单人最佳"都基于 5 条，1 个幸运 case 拉开差距。Windows GPU 跑 N=30 才能拍板
- **FRCRN 的 4-8kHz 填充**听感是否能接受？需要人耳确认
- **业务侧的硬件路线**：16k 真采何时落地？决定我们要不要继续优化 8k 上采路线
- **微调路线已基本否决**（2026-05-08 讨论）：
  - 分离/降噪需要 paired (mixture, clean source) 数据，业务录音拿不到 clean source
  - 8kHz 上采 vs 公开库 16kHz 分布不匹配
  - 自采干净儿童语音库成本 50-100 万 + 半年
  - → 当前阶段重心放在「非训练侧优化」（削波/VAD/参数调优）

---

## 📝 给下一个 Claude session 的"接班"指令

```
我在 [Windows / MacBook] 上和你做这个 ASR 项目, 代码已通过 git 同步.
请按以下顺序快速 catch up:

1. 读 ASR_pipeline_v2/PROGRESS.md (本文件) —— 知道现在在哪
2. 读 ASR_pipeline_v2/README.md G.6/G.7/G.8 章节 —— 知道测试方法
3. 跑 `git log --oneline -10` —— 看最近 commit
4. 简单复述你的理解 (3-5 句), 然后等我下一步指令
```

---

## 🔧 跨机器协作约定

### Git 工作流

```bash
# 开始工作前
cd 公司项目
git pull origin main

# 干活...

# 结束前
git add .
git commit -m "..."
git push origin main
```

**禁忌**：两台机器同时改不同的事——除非你能处理 merge conflict。

### 硬件分工建议

- **Windows (RTX 5090)**：重 GPU 任务
  - 数据清洗、模型推理、批量评估
  - **当前最重要的事：跑 N=30 的 test_audio_methods_compare.py**
- **MacBook (CPU)**：轻量任务
  - 标注、写代码、文档、汇总分析、smoke test (N=5)

### 模型权重

不在 git 里（太大），各机器首次跑相关脚本会自动下载到 `~/.cache/`。

新增依赖：
- `clearvoice` — FRCRN 用 (`pip install clearvoice`)
- 模型路径在 `~/Library/Caches/DeepFilterNet/` 和 ModelScope cache

---

*结束工作前请更新本文件的"最近一次工作" + "下一步" 两节，1 分钟即可。*
