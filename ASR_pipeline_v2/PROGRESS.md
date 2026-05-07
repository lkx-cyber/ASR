# 当前进度

> **作用**：跨机器/跨人协作的"留言板"。每次工作结束前更新，下次开始前先读。
>
> **更新规则**：每个 git push 之前最多花 1 分钟同步本文件。

---

## 📅 最近一次工作

- **时间**：2026-05-08
- **机器**：Windows (RTX 5090)
- **完成**：
  - 全量跑完 `clean_dataset_v2`，3458 条 → 253 条 green++（黄金集）
  - 给老板做了 demo (`sample.zip`，3 多人成功 + 2 单人对照)
  - 写了 `label_groundtruth.py` 标注工具
  - README 大幅更新（v2 实验结论、新工具使用方法）

---

## 🎯 当前状态

### 数据
- `recordings_raw/`: 3458 条原始 PTT 录音 (8kHz, 已 git track)
- `recordings_cleaned/`: v1 清洗，13 信号
- `recordings_cleaned_v2/`: v2 严格清洗，5 个新信号
  - **green++ 253 条** ⭐ 听感全部高质量，可作为评估用黄金集
  - green+ 1826 条（4/5 占绝大多数），听感参差
  - yellow/orange/red: 不达标
- `verify_multi/`: 61 条多说话人样本

### 工具链
- `clean_dataset_v2.py` ⭐ 严格清洗
- `label_groundtruth.py` ⭐ 人工标注（命令行交互）
- `is_multispeaker.py` 单/多人判别
- `find_separation_demo.py` 自动找有展示价值的多人 case
- 各种 test_*.py 实验脚本

### 已验证的结论
- ❌ MossFormer2 在 8kHz 上采数据上电流声，不可用
- ❌ baseline 分离对单人 PTT 有副作用（引入错字）
- ❌ enhance_target 对真重叠场景无效
- ❌ pipeline_router 改善有限
- ✅ **FunASR raw 直转是当前最佳方案**（30/30 召回，0 幻觉）
- ✅ DFN3 降噪边际收益小（FunASR 本身抗噪强）

---

## 🔜 下一步要做的（按优先级）

### 高优先级

- [ ] **标 100 条 ground truth**（用 `label_groundtruth.py`，约 2 小时人工）
  - 输出 `groundtruth.csv`
  - 解锁所有 pipeline 的客观 CER 评估
  - 可以让任何团队成员做（或自己抽空标）

- [ ] **写 `eval_with_groundtruth.py`**（标注完后做）
  - 加载 groundtruth.csv
  - 对每个候选 pipeline 算 CER
  - 输出对比表

### 中优先级

- [ ] **扩展 green++**：探索把 green+ 中"4/5 通过"的 1695 条升级
  - 抽 50 条听感校验
  - 如果 80% 高质量 → 调阈值，黄金集从 253 → 1700+

- [ ] **试 SenseVoice 当主 ASR**（已加载但没作为主路）
  - 替换 Paraformer，对比 CER

### 长期

- [ ] 收集真实业务多说话人录音（更多 8kHz 样本）
- [ ] 探索 8kHz 兼容的 TSE 模型（如 WeSep）

---

## ❓ 待决策 / 阻塞

- **没有 ground truth → 暂时无法客观对比方案优劣**（最大阻塞）
  - 标 100 条解锁
- **业务需求是否扩展？**
  - 当前定位：只要主说话人文本
  - 未来若要"主+背景全部识别"则架构要重做

---

## 📝 给下一个 Claude session 的"接班"指令

```
我在 [Windows / MacBook] 上和你做这个 ASR 项目, 代码已通过 git 同步.
请按以下顺序快速 catch up:

1. 读 ASR_pipeline_v2/PROGRESS.md (本文件) —— 知道现在在哪
2. 读 ASR_pipeline_v2/README.md 末尾的"v2 阶段实验结论" —— 知道哪些路走过了
3. 跑 `git log --oneline -8` —— 看最近 8 个 commit 的变化
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

- **Windows (5090)**：重 GPU 任务
  - 数据清洗、模型推理、批量评估
- **MacBook**：轻量任务
  - 标注、写代码、文档、汇总分析

### 模型权重

不在 git 里（太大），各机器首次跑相关脚本会自动下载到 `~/.cache/`。

---

*结束工作前请更新本文件的"最近一次工作" + "下一步" 两节，1 分钟即可。*
