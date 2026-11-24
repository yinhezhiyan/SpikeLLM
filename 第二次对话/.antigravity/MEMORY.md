# Antigravity 记忆文件 (MEMORY.md)

> 最后更新：2026-11-22（日志 018：第十三轮 8B 回传复核、`datautils.py` 补丁 `f0786e8472931dfe` 出炉与 Codex 012 四项纠偏落实）

## 1. 核心目标、术语铁律与分工边界
- **目标**：在 SpikeLLM (ICLR 2025) 基础上完成 W4A4 量化改进论文（代号 **OSDSQ v2**），冲刺 **ICML 2027**（截稿 `2027-01-22`，备选 `NeurIPS 2027`）。
- **术语铁律**：
  - W4A4 下 90% 低显著通道采用 **1 步 4-bit GIF 神经元**，10% 高显著通道采用 **2 步（`2^addbit` 步）4-bit GIF 神经元**；仅 `SpikeLLM_Ter` 三值权重才实现全累加 AC。
  - Stage 1 定名：**混合步长 GIF 神经元的动态范围耦合失真与子空间隔离校准（Subspace-Isolated Calibration）**；Stage 2 Design (B) 机理定名：**面向真实 FP16 目标的权-激误差协同补偿与块内等 MAC 硬件对齐配额重分配**。
  - **严格禁区（Codex 012 纠偏）**：(a) **严禁**把 GQA `k/v` 未动写成“自动发现的保护机制”（实因 `O_kv << max_O` 使 `dk_kv` 超出可行区间：8B 为 `1148 > 738`，1.1B Tile-32 为 `704 > 576`）；(b) 10% vs 20% 一律表述为“**一半高精度（2-step GIF）通道比例**（名义有效 MAC 系数 `1.10×` vs `1.20×`）”，绝不写成“总算力减半”；(c) 显存必须区分 **`cal_peakVRAM`（校准峰值）** 与 **`eval_peakVRAM`（全模型评测峰值）**；(d) 单层 INT8 计时不能替代整模型 INT4 时延；(e) `SpikeLLM Table 4 (lm-eval-v0.4.2)` 零样本不能与旧 OmniQuant `55.82%` 跨协议相减。

## 2. 核心实验账本（截至 2026-11-22）
### 2.1 LLaMA-2-7B OmniQuant 管线完整三种子主表（实验室 RTX 3090，`seed=2, 3, 4`）
- FP16：`5.4722 / 6.9727` | Original：`10.5806 / 13.7801` | A-off：`10.2818 / 13.0319`
- Stage 1 Full：`8.0370 / 10.7018`（−24.04% / −22.34%）
- **Stage 1+2 AQ**：**`7.5932 ± 0.1123 / 10.1320 ± 0.0103`**（较 Stage 1 再降 **5.52% / 5.32%**，累计降 **28.23% / 26.47%**）
- 7B `seed=2` 等 MAC 三策略：`Uniform 7.9792` | `Random 7.9994` | `Heuristic down/o 7.8057` | **`Adaptive 7.5296`**（较固定规则再降 **3.54%**）。
- 7B AQ `seed=2` 六任务零样本：`55.8246%`（仍待补同协议 `Full s2` / `Orig s2`）。

### 2.2 LLaMA-3-8B OmniQuant 第 13 轮回传（实验室 RTX 3090，`nsamples=64, seed=2`, `WT2` 已出、`C4` 待 20 分钟免重训恢复）
- **WT2 PPL**：`FP16 6.1358` | `Orig 61.5950` | `Stage 1 Full 16.5609`（`−73.11%` vs Orig）| **`Stage 1+2 AQ 14.7334`**（`−11.03%` vs Full，`−76.08%` vs Orig；32 层平均 block MSE `−6.15%`，全层 `high_MACs = 21,827,584`）。
- **复核定性**：(a) `QuantLlamaDecoderLayer` 在 `GQA 4:1 + rope_theta=500000` 关量化下误差 `0.0`，无前向 Bug；`Orig = 61.60` 系 LLaMA-3-8B 过训练离群值 + GQA 跳过 `v→o / q↔k` 两组 LET 平滑 + 无隔离步长拉爆叠加所致；(b) C4 崩于 `main.py:289` (`seqlen=2048`) 触发 `datautils.py:118` off-by-one；(c) 已修 `datautils.py` (`f0786e8472931dfe`) 且本地验证 `--resume omni_parameters.pth --epochs 0` 三腿均为 `max_logit_diff = 0.0`，预计 20 分钟免重训补齐 C4。

### 2.3 `TinyLlama-1.1B`（22L GQA）官方 `spcl/QuaRot-GPTQ + Design (B) + Tile-32` 预演（日志 017，本地 5060 Ti，`cal_peakVRAM = 0.89 GB`）
- **FP16 旋转无损门禁 PASS**：未旋转 `7.7735 / 9.2924` vs 官方 QuaRot（精确 44 阶 Paley）`7.7741 / 9.2928`。
- **纯官方 `QuaRot-GPTQ W4A4`**：`9.3201 / 11.3930`（`k_groupsize=v_groupsize=64`）。
- **`low_p = 0.9`（10% 紧预算）**：`SpikeLLM-QuaRot 9.2166 / 11.2803` | `Stage 1 9.0874 / 11.1519` | `Stage 1+2 (Design B, unaligned) 9.0132 / 11.0556` | **`Stage 1+2 (Design B, Tile-32) 9.0422 / 11.0594`**（用一半高精度通道比例双语料优于 20% `SpikeLLM-QuaRot` `9.1171 / 11.1499`；QuaRot 下 `o_proj +16/-0` 为第一增配对象）。
- **`low_p = 0.8`（20% 宽预算）**：`SpikeLLM-QuaRot 9.1171 / 11.1499` | `Stage 1 8.9229 / 10.9403` | `Stage 1+2 (unaligned / Tile-32) 8.9245 / 10.9342` 与 `8.9409 / 10.9356`。
- **第 14 轮 7B 正式脚本待修 3 处口径**：(1) 拆解 `saliency_mode` 与 `shield`；(2) `Stage 1 (Tile-32, search OFF)` vs `Stage 1+2 (Tile-32, search ON)` 使用完全相同的起始配额与 mask 规则；(3) 分开打印 `cal_peakVRAM` 与 `eval_peakVRAM`。

### 2.4 端到端硬件 Tile 微基准（日志 017，RTX 5060 Ti，7B 单层高显著分支 `L=2048`）
- 端到端真实 INT8 (`F.pad + torch._int_mm`)：R11 未对齐 `1.8461 ms` → `Aligned-8 (1.7480)` / `16 (1.7566)` / **`32 (1.7828, +3.6%)`** / `64 (1.8717, −1.4%)`；FP16 `Aligned-32 (1.7791 ms)` 最优 → **第 14 轮锁定 `--quota_tile_size 32`**。

## 3. 代码库当前哈希
- [datautils.py](file:///d:/spikellm/SpikeLLM/datautils.py)：**`f0786e8472931dfe`** (`6,728 B`，第 13 轮 C4 恢复唯一需同步补丁，仅改 103/118/167 行)
- [main.py](file:///d:/spikellm/SpikeLLM/main.py)：`bb23faf27f7b3688` (`19,508 B`，含 `--quota_tile_size`，默认 0)
- [spike_driven_quant/spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py)：`240fd2ba99829d1b` (`25,844 B`)
- [spike_driven_quant/spike_linear.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_linear.py)：`857be868ad9cc740` (`9,487 B`)
