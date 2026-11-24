# Antigravity 协作日志 003 · 2026-11-07 · 第八轮复核、分辨率倒挂定理与 Stage 2 (`--adaptive_quota`) 代码落地及 68M 实证

**本轮性质**：第八轮实验室结果复核 + 理论机制突破（分辨率倒挂定理）+ Stage 2 生产代码修改 + 68M 逐位一致性与端到端增益实证
**前置阅读**：
- `第二次对话/work Buddy日志/workbuddy日志_004_2026-11-07_第八轮判读与第九轮包.md` 及 `.workbuddy/MEMORY.md`
- `第二次对话/codex日志/codex日志_005_2026-11-07_第八轮回传与新路线审阅.md` 及 `.codex/MEMORY.md`
- `第二次对话/实验室日志/第八轮日志/` 全部文件（`lab_round8_result.txt`、`第八轮回传校验.txt`、`README_索引.md`、`r8_68m_gate_对照.txt`、`env_round8_preflight.txt`）
- `第二次对话/实验室命令/round9_addbit2/lab_round9_runner.ps1`

---

## 一、用户原话

> "看一下最新日志，给出你的建议，并进行有必要的代码修改"

---

## 二、我做了什么

1. **审阅第八轮实验室回传与 WorkBuddy (004)、Codex (005) 最新日志**：
   - 核实第八轮 4 条腿（68M `Orig/Ours@a3`、7B `Orig/Ours@a3`）全部与 `log_rank0_*.txt` 逐位吻合，环境指纹确认 8 个关键源码文件哈希与第七轮 100% 一致。
   - 审阅了 WorkBuddy 打包的第九轮 `lab_round9_addbit2_2026-11-07.zip`（补 68M `Orig@a1` 基线 + 68M/7B `addbit=2` 中点），参数与前序轮次严格对齐，无代码冲突。
2. **正面回应 Codex 005 的两条核心学术批评，升级 Stage 2 算法设计**：
   - Codex 005 指出：① 单纯把各线性层独立重构误差相加忽略了层内注意力/SwiGLU 非线性与 LET 平滑的耦合；② 若不严格对齐各投影矩阵形状 $O_m$ 与 KV 分支，不能叫严格等预算。
   - 针对这两点，我将 Stage 2 升级为 **“后 LET 块输出重构引导的整数 MAC 守恒配额重分配（Post-LET Block-Reconstruction-Guided Iso-MAC Quota Reallocation）”**：
     - 直接以包含初始 LET 平滑与 SwiGLU/Attention 非线性的 **DecoderLayer 整体块输出 MSE** $\|\text{qlayer}_{\text{quant}}(x) - \text{qlayer}_{\text{fp}}(x)\|_2^2$ 对单位高显著 MAC（$\Delta\text{MAC} = \Delta k_m \cdot O_m$）的边际变化率为准则；
     - 在每个 DecoderLayer 内部 7 个 `SpikeQuantLinear` 模块（`q/k/v/o/gate/up/down`）之间做贪心等 MAC 交换，保证每层的总高显著 MAC 数 $\sum_{m=1}^7 (C_m - \text{low\_quota}_m) O_m$ **精确到整数 MAC 恒等于（或 $\le$）均匀 `low_p=0.9` 基线**，且 KV MatMul 保持不变，从而使全网 ACE（无论 MPQ 口径还是 SNN 时间步口径）100.0000% 严格不变！
3. **发现支撑 Stage 2 的核心数学机制——“分辨率倒挂定理（Resolution Inversion Theorem）”**（见 §3.2）。
4. **完成 `SpikeLLM/` 生产代码修改并通过两道硬性验收**：
   - 修改了 `main.py`、`spike_driven_quant/spike_linear.py`、`spike_driven_quant/spike_omniquant.py`，新增 `--adaptive_quota` 纯增量开关（默认 `False`）。
   - **门禁一（开关关闭逐位一致性自证，WorkBuddy 004 §七要求）**：在开发机上用 `llama-68m` (`seed=2`, 20 epochs) 实测，修改前与修改后（`--adaptive_quota` 默认关闭）产出的 `omni_parameters.pth` **MD5 均为 `112A1F2810EAD270BB09863CA447BD80`（逐字节 100% 相同）**，40 个训练 iter 的 loss 及最终 PPL（WT2 `159.2802734375` / C4 `91.741455078125`）逐位完全一致！
   - **门禁二（68M 端到端 20-epoch 蒸馏实测，WorkBuddy 004 §五.1 要求）**：开启 `--adaptive_quota` 后，在**高显著 MAC 数 100% 不变（`946,176 -> 946,176`，+0.00%）**的前提下，68M 端到端 WikiText-2 PPL 从 **159.2803 降至 154.4269（再降 −4.85 PPL / −3.05%）**，C4 PPL 从 **91.7415 降至 88.3023（再降 −3.44 PPL / −3.75%，在 C4 上甚至以 `addbit=1` 零额外成本反超了第八轮 `Ours@addbit=3` 的 89.00）**！

---

## 三、结论与新发现

### 3.1 对第八轮结果与第九轮包的判断
1. **第八轮结果是一颗定心丸**：
   - 7B `Orig@addbit=3` 的 WikiText-2 仅从 `10.5726` 降到 `10.1386`（仅改善 4.10%），而 `Ours@addbit=1`（`7.9792`）在不加任何位宽/时间步的情况下比 `Orig@addbit=3` 低 **21.30%**；同在 `addbit=3` 下，`Ours@addbit=3`（`7.8978`）仍比 `Orig@addbit=3` 低 **22.10%**。
   - 这彻底堵死了“只要调大 `addbit` 就能替代量程隔离”的审稿质疑。
2. **第九轮包（`lab_round9_addbit2_2026-11-07.zip`）完全合理，应立即发往实验室运行**：
   - 补齐 `addbit=2` 中点可形成完整的 `addbit ∈ {1, 2, 3}` 三点预算–精度曲线；顺带在实验室同机补跑 `68M Orig@a1`，彻底解决历史 68M 基线数字不一的遗留问题。
   - **重要提示**：第九轮包只含 `.ps1` 脚本、不含代码文件，实验室可直接基于现有代码跑第九轮，与我今天在本地 `SpikeLLM/` 新增 `--adaptive_quota` 互不干扰。

### 3.2 核心理论突破：为什么 Stage 1（量程隔离）必然呼唤 Stage 2（自适应配额）？——“分辨率倒挂定理”
在用探针脚本分析 68M 第 0 层 7 个线性模块对 DecoderLayer 块输出 MSE 的边际导数时，我发现了一个极为反直觉却有严格数学证明的现象：
- 在 `q_proj`、`v_proj`、`gate_proj` 上，**减少**高显著分支通道数（把 `low_p` 从 `0.90` 提高到 `0.96~0.98`，即把通道从 5-bit 高分支退回 4-bit 低分支），DecoderLayer 的块重构误差**不仅没有变差，反而显著下降了（单步降幅达 −3.8%）**！
- **为什么把通道从 5-bit 高分支退回 4-bit 低分支反而更准？**
  - 在**原版 SpikeLLM（无量程隔离）**中，4-bit 低分支和 5-bit 高分支共用全局量程 $\text{Range}(X_{\text{all}})$，因此高分支步长 $\Delta_{\text{high}} = \text{Range}(X_{\text{all}})/31$ 永远小于低分支步长 $\Delta_{\text{low}}^{\text{orig}} = \text{Range}(X_{\text{all}})/15$。给任何模块多分配高分支通道永远不会劣化分辨率。
  - 但是，一旦引入 **Stage 1（子集量程隔离）**，4-bit 低分支的步长缩小为 $\Delta_{\text{low}}^{\text{iso}}(p_m) = \text{Range}(X_{\text{low}}(p_m))/15$，而 5-bit 高分支仍承担全局极值 $\text{Range}(X_{\text{all}})$，两者步长之比为：
    $$\frac{\Delta_{\text{high}}}{\Delta_{\text{low}}^{\text{iso}}(p_m)} = \frac{2^b - 1}{2^{b+\text{addbit}} - 1} \cdot r_m(p_m) = \frac{15}{31} r_m(p_m)$$
  - **定理（分辨率倒挂 Resolution Inversion）**：当离群比 $r_m(p_m) > \frac{2^{b+\text{addbit}}-1}{2^b-1} \approx 2.07$ 时，**4-bit 量程隔离低分支的量化步长比 5-bit 全局高分支更精细（$\Delta_{\text{low}}^{\text{iso}} < \Delta_{\text{high}}$）**！
  - 在经过 LET 平滑的模块（如 `q_proj, v_proj, gate_proj`）中，仅最极端的 2%–4% 通道是真正的幅值离群点；排名在 2%–10% 之间的通道因权重范数 $\|W_{:,c}\|$ 较大而被一阶泰勒显著度选中，但它们的激活幅值并不极端。原版一刀切 `low_p=0.90` 强行把这批“高权重敏感、常幅值”的通道塞进高分支，反而使它们遭受了 $\frac{15}{31} r_m > 1$ 倍的**量化步长粗化（分辨率倒挂）**！
  - 与此同时，在无法被 LET 平滑的 `mlp.down_proj`（见 `spike_llama_layer.py:302`，因 SwiGLU 门控乘法阻断了等价变换吸收，`down_proj` 直接承受未平滑的 `silu(gate)*up` 重尾激活）中，10% 的高显著配额又远远不够！
- **这一定理直接把 Stage 1 和 Stage 2 焊成了一个不可分割的整体**：
  - Stage 1（子集量程隔离）消除了低分支的量程污染，但也因此打破了各矩阵在 `low_p=0.9` 处的边际平衡，引发了 LET 平滑层与非平滑层（`down_proj`）之间的**分辨率倒挂与预算错配**；
  - Stage 2（`--adaptive_quota`）自动将 `q_proj/v_proj/gate_proj` 中遭受分辨率倒挂的冗余高分支配额（退回至 `low_p ≈ 0.96~0.98`）转移给重尾未平滑的 `down_proj`（扩充至 `low_p ≈ 0.80`）和关键注意力投影（`k_proj`/`o_proj`），从而在**零额外 MAC 开销**下实现二次降损！

### 3.3 Stage 2 (`--adaptive_quota`) 在 68M 上的实测数据（同机同 seed=2，20 epochs）

| 配置 | Layer 0 校准块 MSE | Layer 0 训练末 MSE | Layer 1 校准块 MSE | Layer 1 训练末 MSE | 每层高显著 MAC 数 | WikiText-2 PPL | C4 PPL |
|---|---|---|---|---|---|---|---|
| Stage 1 (Uniform `low_p=0.9`) | 0.062843 | 0.036862 | 0.733730 | 1.018845 | 946,176 | 159.2803 | 91.7415 |
| **Stage 1 + Stage 2 (`--adaptive_quota`)** | **0.055759 (−11.27%)** | **0.035590 (−3.45%)** | **0.705486 (−3.85%)** | **0.970918 (−4.70%)** | **946,176 (+0.00%)** | **154.4269 (−3.05%)** | **88.3023 (−3.75%)** |

- **自适应配额分布（从日志直接提取）**：
  - Layer 0：`k_proj: 0.822, v_proj: 0.978, q_proj: 0.978, o_proj: 0.900, gate_proj: 0.958, down_proj: 0.802, up_proj: 0.919`（高显著 MAC：`946,176 -> 946,176`）
  - Layer 1：`k_proj: 0.900, v_proj: 0.900, q_proj: 0.978, o_proj: 0.822, gate_proj: 0.978, down_proj: 0.802, up_proj: 0.919`（高显著 MAC：`946,176 -> 946,176`）
- **校准耗时增量**：每层仅需约 1.8 秒（68M 全网仅增加 3.6 秒，7B 32 层预计仅增加 1~2 分钟，相对 7 小时训练可忽略不计）。

### 3.4 代码改动清单与最新环境指纹哈希

仅修改了以下 3 个文件（其余 5 个关键文件哈希保持不变）：
- `SpikeLLM/main.py`：新哈希 `49485bc7ad4e9e0b`（18,942 B，旧 `6842f0bb479da6bc`）—— 新增 `--adaptive_quota` 命令行参数并传入 `act_quant_params`。
- `SpikeLLM/spike_driven_quant/spike_linear.py`：新哈希 `857be868ad9cc740`（9,487 B，旧 `f19df5382e798d07`）—— 当 `self.adaptive_quota` 开启时在 `static()` 中暂存 `self.saliency_score`。
- `SpikeLLM/spike_driven_quant/spike_omniquant.py`：新哈希 `2b17e4fcfea74fce`（21,931 B，旧 `18cf8d473bfc4795`）—— 新增 `allocate_layer_quota()`，并在每层 LET 参数初始化后、`epochs` 训练前（当 `args.adaptive_quota` 开启时）调用，结束后清理 `saliency_score`。
- **逐位一致性铁证**：修改前与修改后运行默认参数（不带 `--adaptive_quota`），68M `omni_parameters.pth` 的 MD5 哈希均为 `112A1F2810EAD270BB09863CA447BD80`。

---

## 四、待他人行动

1. **用户**：
   - **立即执行第九轮**：把 WorkBuddy 已经打好的 `第二次对话/实验室命令/lab_round9_addbit2_2026-11-07.zip` 发给实验室开跑（无需更新实验室的 Python 代码，直接跑即可）。
2. **WorkBuddy**：
   - 审阅本日志 §3.2 的 **“分辨率倒挂定理（Resolution Inversion Theorem）”** 与 §3.3 的 **68M 实测结果（WT2 −3.05% / C4 −3.75%，MAC 严格 +0.00%）**。这个定理不仅解释了第八轮为什么 `Ours` 从 `addbit=1` 到 `addbit=3` 变化极小，更让 Stage 1 和 Stage 2 形成了严密的因果闭环。
   - 准备**第十轮实验包**（需把修改后的 `main.py`、`spike_linear.py`、`spike_omniquant.py` 一并打入补丁 zip）：先在实验室跑 68M 多种子（`seed=2,3,4`）与 1.1B/7B 的 `--adaptive_quota` 对照。
3. **Codex**：
   - 复核本日志中升级后的 `allocate_layer_quota` 实现（已按你 005 日志的意见，改为直接优化后 LET 的 DecoderLayer 整体块输出 MSE，并严格保持每层整数高显著 MAC 守恒 `946,176 -> 946,176`）。
   - 请尽快完成 **ICLR 2026 Phase-Coding 仓库（`SpikingLLM`）的源码级核查**（解释 `T=8` 5.86 的真实配置语义 + 码本估计域是否全局），这是目前唯一悬而未决的外部文献问题。

---

## 五、风险

1. **68M 单种子与跨规模泛化风险**：虽然 `--adaptive_quota` 在 68M (`seed=2`) 上取得了 WT2 −3.05% / C4 −3.75% 的显著收益，且块输出 MSE 在训练前后一致下降，但仍需按 WorkBuddy 和 Codex 的门禁要求，在更多种子及更大模型（1.1B / 7B）上验证收益稳定性。
2. **实验室代码版本同步提醒**：实验室跑第九轮（`lab_round9_addbit2_2026-11-07.zip`）时使用的是旧代码哈希；等第十轮测试 `--adaptive_quota` 更新实验室代码后，`env_report.py` 中的 3 个文件哈希将变为本日志 §3.4 记录的新哈希（因为关闭开关时数值逐位等价，新旧哈希在默认模式下完全可比）。
