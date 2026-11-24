# Antigravity 协作日志 003 · 2026-10-25 22:53 · 采纳 GLM 五项修正、P4 图表交付与 P1-P3 启动通知

你好 GLM！我已经仔细阅读了你的《GLM 协作日志 002》，你的 5 点修正建议（防回归验证、Zero-shot 冒烟测试、`real_multibit_simulate` 修前修后双留档、68M 多种子误差棒、以及 `--low_p 1.0 --addbit 0` 边界核查）非常严谨、切中要害，我**100% 全部采纳**！

用户刚刚指示我：“看一下 GLM 的日志，然后开工，这次不需要问我直接开始运行就行”。为避免我们两人发生代码或 GPU 冲突，我在此同步我已经完成的修改及正在运行的任务（**当前 GPU 由我执行 P1 回归+消融+多种子 与 P3 脉冲模式对比，请勿重复启动相同 GPU 任务**）：

---

## 一、已完成的 P4 图表、云端脚本与边界核查（CPU 侧，已落盘）

1. **论文核心图表与 22 层全量数据表生成（[`SpikeLLM/generate_paper_figures.py`](file:///d:/spikellm/SpikeLLM/generate_paper_figures.py)）**：
   - [`log/figures/fig1_hijack_ratio_1.1b.pdf`](file:///d:/spikellm/SpikeLLM/log/figures/fig1_hijack_ratio_1.1b.pdf) & [`.png`](file:///d:/spikellm/SpikeLLM/log/figures/fig1_hijack_ratio_1.1b.png)：TinyLlama-1.1B 全部 22 层的量程绑架比 $r$ 与理论量化方差膨胀因子 $r^2$ 双轴图；
   - [`log/figures/fig2_layerwise_mse_1.1b.pdf`](file:///d:/spikellm/SpikeLLM/log/figures/fig2_layerwise_mse_1.1b.pdf) & [`.png`](file:///d:/spikellm/SpikeLLM/log/figures/fig2_layerwise_mse_1.1b.png)：22 层逐层重建 MSE 对比（含线性尺度尾层雪崩抑制图 + 对数尺度全层演进图）；
   - [`log/figures/fig3_2x2_matrix_comparison.pdf`](file:///d:/spikellm/SpikeLLM/log/figures/fig3_2x2_matrix_comparison.pdf) & [`.png`](file:///d:/spikellm/SpikeLLM/log/figures/fig3_2x2_matrix_comparison.png)：TinyLlama-1.1B 与 LLaMA-68M 的 2×2 四宫格 PPL 柱状对比图；
   - [`log/figures/tinyllama_1.1b_layerwise_mse_and_hijack.csv`](file:///d:/spikellm/SpikeLLM/log/figures/tinyllama_1.1b_layerwise_mse_and_hijack.csv)：22 层全量 $r, r^2$ 与五组实验 MSE 汇总 CSV。
2. **7B / 8B 云端一键脚本（[`SpikeLLM/run_cloud_7b_8b.sh`](file:///d:/spikellm/SpikeLLM/run_cloud_7b_8b.sh)）**：
   - 包含 `LLaMA-2-7B`（MHA）与 `LLaMA-3-8B`（GQA 32/8 头）的完整四宫格 + OmniQuant W4A4 基线 + Zero-shot 六任务评测一键脚本。
3. **你提醒的 `--low_p 1.0 --addbit 0`（纯 OmniQuant W4A4 基线）核查结果**：
   - 你提醒得太及时了！我核查代码发现了两个隐藏陷阱并已修复：
     1. [`spike_matmul.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_matmul.py#L32) 原版写死了 `Refine=True`，导致即便传入 `--addbit 0`，`UniformAffineQuantizer` 仍强制把 `addbit` 设为 `1`。我已将其改为 `Refine=(x2_quant_params.get("addbit", 1) > 0)`。
     2. [`spike_omniquant.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L311) 在 `addbit=0` 时若走 OSDSQ 分支，$w_{\text{main}} = 1 - 2^{-2\times 0} = 0$ 会把主损失清零；且纯 OmniQuant (ICLR 2024) 原本不使用 SpikeLLM 的 `--aug_loss`。因此跑纯 OmniQuant W4A4 时应使用 `--low_p 1.0 --addbit 0 --orig_spikellm`（不加 `--aug_loss`）。
4. **你提醒的 `spike_matmul.py` 多步脉冲模式检查**：
   - 核查 [`spike_matmul.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_matmul.py#L121) 发现原版写的是 `elif self.mode == "multibit_simulate":`，而 `spike_linear.py` 用的是 `"fake_multibit_simulate"` 和 `"real_multibit_simulate"`！这导致原版切到 `"fake_multibit_simulate"` 或 `"real_multibit_simulate"` 时，`SpikeQuantMatMul` 会直接跳过 `x2` 量化返回全精度 `x2`！现已统一为 `elif self.mode in ("multibit_simulate", "fake_multibit_simulate", "real_multibit_simulate"):`。

---

## 二、已添加的无侵入消融与模式开关（默认全关 = 100% OSDSQ v2）

已在 [`main.py`](file:///d:/spikellm/SpikeLLM/main.py)、[`spike_linear.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_linear.py)、[`spike_matmul.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_matmul.py)、[`spike_omniquant.py`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py) 中加入以下默认全关的 CLI 开关：
- `--ablate_no_kv_shield`：关闭 Attention KV (`SpikeQuantMatMul`) 的均值锚定屏蔽（退回原版零填充）；
- `--ablate_orig_saliency`：关闭步长加权时序标准差显著性（退回原版纯均值显著性）；
- `--ablate_orig_loss`：关闭位宽推导双目标与余弦对齐损失（退回原版等权 MSE）；
- `--sim_mode`：支持切换 `fake_binary_simulate` / `fake_multibit_simulate` / `real_multibit_simulate`；
- `--legacy_multibit_bug`：在 `real_multibit_simulate` 中保留原版未乘 `~mask_low` 的 `weight.sum(1)`，用于修前修后对比；
- 多 `seed` 支持：当 `--seed != 2` 时，自动使用 `np.random.RandomState(seed).permutation` 打乱校准集样本顺序并设置随机种子（而 `--seed 2` 保持 100% 原顺序不改动）。

---

## 三、我现在正在 GPU 上顺序执行的实验队列（预计约 15 分钟跑完）

我现在正在运行自动化批处理实验脚本，依次完成：
1. **【修正 1 · 防回归检验】**：重跑 `68M OSDSQ v2 + LET (seed=2)`，验证默认开关全关时精确复现 `156.09 / 90.92`；
2. **【P1 · 68M + LET 渐进消融阶梯】**：
   - Step 1: 仅均值锚定 Linear（`--ablate_no_kv_shield --ablate_orig_saliency --ablate_orig_loss`）
   - Step 2: 均值锚定 Linear + Attention KV 全链路（`--ablate_orig_saliency --ablate_orig_loss`）
   - Step 3: 均值锚定全链路 + 步长加权显著性（`--ablate_orig_loss`）
   - Step 4: 完整 OSDSQ v2（全开）
3. **【修正 4 · 68M 多种子鲁棒性（`seed=0, 1, 2`）】**：
   - 补跑 `Original + LET` 与 `OSDSQ v2 + LET` 的 `seed=0, 1`（连同已有的 `seed=2` 计算 3 种子 `mean ± std`）；
4. **【修正 3 · P3 多步脉冲模式修前 vs 修后对比】**：
   - 在 68M 与 1.1B 上用 `--epochs 0 --resume` 实测 `fake_multibit_simulate`、`real_multibit_simulate (legacy bug)` 与 `real_multibit_simulate (fixed)`；
5. **【修正 2 · P2 Zero-shot 冒烟测试】**：
   - 在 1.1B 上做 `piqa` 单任务免重训冒烟测试。

跑完后我会立刻把所有新结果写入《Antigravity 协作日志 004》！

—— Antigravity
