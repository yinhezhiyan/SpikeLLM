# Antigravity 日志 018：第十三轮 8B 深度复核、`datautils.py` 补丁出炉与 Codex 012 纠偏全面落实（2026-11-22）

## 一、用户原话
> 看一下最新日志

---

## 二、我做了什么
1. **前置通读全部最新文档与实验室回传**：通读 [workbuddy日志_010_2026-11-22_第十三轮判读与8B_c4边界bug.md](file:///d:/spikellm/第二次对话/work Buddy日志/workbuddy日志_010_2026-11-22_第十三轮判读与8B_c4边界bug.md)、[codex日志_012_2026-11-22_8B回传与QuaRot预算复核.md](file:///d:/spikellm/第二次对话/codex日志/codex日志_012_2026-11-22_8B回传与QuaRot预算复核.md)，以及 `实验室日志/第十三轮日志/第十三轮回传/` 下全部文件（`给开发组的说明_c4失败与修法.md`、`lab_round13_result.txt`、`LAB_ROUND13_CONFIG.txt`、`env_round13_preflight.txt`、`r13_c4_diag.txt` 与四腿 `rank0` / `failed` 日志）。
2. **完成 3 项本地实证代码核查**：
   - **Llama-3 GQA + `rope_theta=500000` 前向等价核验**：在本地模拟 `GQA 4:1 + rope_theta=500000.0` 配置，实测 `QuantLlamaDecoderLayer`（量化关闭）与原生 HuggingFace `LlamaDecoderLayer` 的前向输出完全一致（`max_abs_diff = 0.0`）。
   - **8B 三腿逐层重构轨迹与 GQA 行为核查**：解析全部 32 层 `r13_8b_{orig,full,aq}` 终值 loss、`norm:nan` 事件与 32 层 `[adaptive_quota]` 配额流向，定位 `Orig=61.5950` 的真实成因。
   - **`--resume --epochs 0` 检查点恢复位一致性实测**：本地构造 2 层 GQA Llama-3 模型，完整跑通 `orig / full / aq` 三腿 `train(epochs=2) -> save omni_parameters.pth -> fresh model --resume --epochs 0`，验证三腿恢复后 logits 与原训练结果 **100% 位一致（`max_logit_diff = 0.000000e+00`）**，且验证 `torch.load(..., weights_only=True)` 可正常加载 `OrderedDict` 结构（兼容实验室 PyTorch 2.6 默认行为）。
3. **落盘 `datautils.py` 精确 3 处 C4 边界补丁**：遵照 WorkBuddy 010 方案 A 与 Codex 012 安全边界约束，仅修复 [datautils.py](file:///d:/spikellm/SpikeLLM/datautils.py) 中前置已保证 `L >= seqlen` 的 3 处 C4 采样点（第 103、118、167 行），通过 `py_compile` 与边界单测，输出新哈希 `f0786e8472931dfe`。
4. **全面吸收并落实 Codex 012 的 4 项硬核纠偏**，同步更新 [MEMORY.md](file:///d:/spikellm/第二次对话/.antigravity/MEMORY.md)。

---

## 三、结论

### 1. `datautils.py` C4 off-by-one 补丁定案与《实验室补丁同步清单》
- **根因更正（采纳 Codex 012）**：四腿崩溃触发于 [main.py](file:///d:/spikellm/SpikeLLM/main.py#L289) 显式覆盖的 **`lm.seqlen = 2048`**（并非 `max_position_embeddings = 8192`，因此实验室用 `8192` 做的独立诊断返回 `crash_at = None`）。在 `seqlen = 2048` 下，LLaMA-3-8B 128K tokenizer 对某条 C4 validation 样本切词恰好等于 `2048`，通过第 116 行 `>= seqlen` 后在第 118 行产生上界 `-1` 抛出 `ValueError: empty range for randrange() (0, 0, 0)`。
- **精确修法（3 处，不误伤未检查路径）**：
  - 仅修改 `datautils.py` 第 **103、118、167 行**（`get_c4` train、`get_c4` val、`get_c4_new` train，这三处前一行均显式保证 `input_ids.shape[1] >= seqlen`）：`random.randint(0, max(0, ...shape[1] - seqlen - 1))`；
  - 遵照 Codex 012 提醒，**不修改**第 31、53、75、139 行（整段语料拼接路径，无前置长度检查），避免未来误传短语料时静默接受 `< seqlen` 短样本。
- **对既有 7B / 68M 结果零影响论证**：
  1. 7B 历史批次全部命中预生成缓存 `./cache/testloader_Llama_c4_all.cache`，从未进入 `get_c4`；
  2. 对任何 `L > seqlen` 样本，`max(0, L - seqlen - 1) == L - seqlen - 1`，单测验证随机数流与抽样起点 **100% 位一致**；仅当 `L == seqlen` 时避免崩溃并确定性取 `i = 0`（整段长度仍精确等于 `seqlen`）。
- **与实验室代码版本零冲突**：补跑第 13 轮 8B C4 时，实验室**只需覆盖 `datautils.py` 单个文件**，无需改动实验室的 `main.py (e898ff1da6b92e91)` 与 `spike_omniquant.py (4f7ae925a3c1bd51)`，保持与第 13 轮完全同版！

#### 《实验室补丁同步清单》

| 相对路径 | 补丁前 SHA-256 前 16 位 | 补丁后 SHA-256 前 16 位 | 字节数 | 说明 |
| :--- | :---: | :---: | :---: | :--- |
| [datautils.py](file:///d:/spikellm/SpikeLLM/datautils.py) | `54bbaae1c48888e2` | **`f0786e8472931dfe`** | `6,728 B` | **第 13 轮 C4 恢复唯一需同步文件**（仅改 103/118/167 行 C4 边界） |
| [main.py](file:///d:/spikellm/SpikeLLM/main.py) | `e898ff1da6b92e91` | `bb23faf27f7b3688` | `19,508 B` | 本地开发版（新增 `--quota_tile_size`，默认 0）；**第 13 轮恢复暂勿同步** |
| [spike_driven_quant/spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py) | `4f7ae925a3c1bd51` | `240fd2ba99829d1b` | `25,844 B` | 本地开发版（支持 `tile` 对齐）；**第 13 轮恢复暂勿同步** |
| [spike_driven_quant/spike_linear.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_linear.py) | `857be868ad9cc740` | `857be868ad9cc740` | `9,487 B` | 未变更 |

---

### 2. 第 13 轮 8B C4 可在 ~20 分钟内免重训恢复（已本地验证位一致）
- 核查确认：第 13 轮中 [spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L568) 在 `eval_ppl` 之前已逐层落盘 `./log/r13_8b_{orig,full,aq}/omni_parameters.pth`（包含全部 `smooth_scale/shift`、`bound_factor` 及 `mask_low_mse` buffer），且 `lab_round13_runner.ps1` 失败时未清理该文件。
- 本地实测：`orig / full / aq` 三腿通过 `--resume <omni_parameters.pth> --epochs 0 --eval_ppl` 恢复后，与原训练模型前向 logits **`max_logit_diff = 0.000000e+00`**，且兼容 PyTorch 2.6 `weights_only=True`。
- **推荐恢复验证顺序（严格对齐 Codex 012）**：
  1. 替换 `datautils.py` (`f0786e8472931dfe`) 后，先跑 `r13_8b_fp16`（~4 分钟），以 `seqlen=2048` + 8B tokenizer 落盘 `./cache_llama3_8b/testloader_Llama_c4_all.cache`，并记录其 tensor shape `(1, 524288)` 与 SHA-256；
  2. 检查 `./log/r13_8b_{orig,full,aq}/omni_parameters.pth` 是否均完整包含 `0..31` 层键与 `mask_low_mse` buffer；
  3. 对完整检查点跑 `--resume ... --epochs 0 --eval_ppl`（每腿 ~5 分钟），**先验恢复后的 WT2 PPL 与原值 `61.5950 / 16.5609 / 14.7334` 完全一致**，确认无漂移后再将 C4 PPL 正式入账（若某腿文件不完整再单独重训该腿）。

---

### 3. 8B `Orig = 61.5950` 异常深度复核结论（非前向 Bug，系三重机制叠加）
1. **排除 GQA / RoPE 前向包装 Bug**：本地验证 `QuantLlamaDecoderLayer`（关量化）在 `GQA 4:1 + rope_theta=500000` 上误差为 **`0.0`**；实验室 `Full (16.56)` 与 `AQ (14.73)` 均语义正常（若 RoPE `theta` 或 GQA 索引出错，全线均会崩至 >1000）。
2. **造成 `Orig = 61.60` 且 `AQ = 14.73 (2.40× FP16)` 的真实三重机制**：
   - **架构固有难度**：`LLaMA-3-8B`（15T token 过训练 + 128K 词表）在公开量化文献中对无旋转 W4A4 OmniQuant 极度敏感，纯 OmniQuant W4A4 在 8B 上普遍恶化至数十～数百乃至 NaN；
   - **GQA 使 OmniQuant 4 组 LET 平滑跳过 2 组**：[transformation.py](file:///d:/spikellm/SpikeLLM/models/transformation.py#L45-L70) 中因 `v_proj (1024) != o_proj (4096)` 与 `q_proj (4096) != k_proj (1024)`，上游代码跳过 `v→o` 与 `q↔k` 平滑，导致 `o_proj` 输入保留未经平滑的巨型离群值（这直接体现在 8B AQ 32 层统计中：`o_proj` 在 **`27/32` 层**抢占高显著配额，几乎与 `down_proj` **`29/32` 层**持平！）；
   - **`--orig_spikellm` 无隔离叠加误差级联**：关闭子空间隔离后，巨型离群值直接拉爆低显著分支步长；从 Layer 20 起 `Orig` 重构误差显著反超（Layer 31 iter 19：`Orig 0.5483` vs `Full 0.2417` vs `AQ 0.2308`，且 Layer 1 iter 0 出现 1 次 `norm:nan`），最终推高至 `61.60`；开启 Stage 1 后立刻回落 `−73.11%` 至 `16.56`，Stage 2 再降 `−11.03%` 至 `14.73`（32 层平均 block MSE `−6.15%`，全层 `high_MACs = 21,827,584` 严守等算力）。
3. **论文定位（对齐 Codex 012）**：`Orig` 是原版 SpikeLLM 完整基线配置（非单变量消融），且 OmniQuant 下 8B AQ 仍为 FP16 的 `2.40×`；因此 8B OmniQuant 适合作为**跨架构（GQA + 128K 词表）单种子验证证据**，真正对标 2027 顶会 SOTA 的主力绝对分数由 **第 14 轮官方 `spcl/QuaRot-GPTQ` 管线**扛起！

---

### 4. 全面接受并落实 Codex 012 的 4 项硬核纠偏（彻底封堵论文审稿漏洞）
1. **正式撤回“GQA K/V 自动受保护”误读（重大纠错！）**：
   - 核实 Codex 算账 100% 正确：在 GQA 模型上因 `O_kv << max_O`，8B 的 `dk_kv = 82×14336/1024 = 1148 > 738`（可行低通道区间 `[3276, 4014]` 宽度），1.1B `Tile-32` 的 `dk_kv = 704 > 576`（可行高通道区间 `[32, 608]` 宽度），导致 `k_proj / v_proj` **因步长超界从未进入过候选池**（仅 MHA 7B 上 `dk_kv = 220` 正常参与竞争）。
   - 撤回日志 007 与日志 017 中“Stage 2 自动保护 GQA k/v”的错误解读，后续论文仅如实描述观察到的 `o/down` 增配与 `q/gate` 让渡，不把步长约束包装成“自动发现的保护机制”。
2. **第 14 轮 7B 官方 `QuaRot-GPTQ` 正式脚本锁死 3 处口径（修补 `run_1.1b_quarot_design_b.py` 临时脚本缺陷）**：
   - **第 203 行**：将 `saliency_mode` 与 `shield` 拆成独立开关；紧预算/宽预算主表同时保留 `Pure QuaRot-GPTQ` 锚点、`SpikeLLM-QuaRot` 原版基线，以及 **同 GPTQ 权重、同 `Tile-32` 起始配额、同 mask 规则的 `Stage 1 (Tile-32, search OFF)` vs `Stage 1+2 (Tile-32, search ON)`** 干净主对照（消除第 207 行仅对搜索分支向下取整带来的初始高通道差——这也解释了 1.1B `low_p=0.8` 时 `Tile-32` 微涨 `0.018` 的来源）；
   - **第 316 行**：将显存统计拆分为 **`cal_peakVRAM`（逐层校准峰值，1.1B 为 `0.89 GB`）** 与 **`eval_peakVRAM`（全模型评测峰值）** 分别记录，绝不混写。
3. **算力与时延表述严守红线**：
   - 比较 10% (`low_p=0.9`) vs 20% (`low_p=0.8`) 时，一律表述为“**一半的高精度（2-step GIF）通道比例**（名义 4-bit 有效 MAC 系数 `1.10×` vs `1.20×`）”，绝不写成“总算力减半”；
   - 单层 `Tile-32` INT8 实测严禁等价于整模型 INT4 端到端时延。
4. **零样本协议不可跨版本相减**：
   - `SpikeLLM ICLR 2025` Table 4（`lm-eval-v0.4.2`：`QuaRot-GPTQ 65.61%`、`20% SpikeLLM-QuaRot 66.76%`）仅可与第 14 轮同版本协议横向比较，严禁与 OmniQuant 管线旧 `55.82%` 跨协议相减；7B OmniQuant 同协议 `Stage 1 Full s2` 与 `Orig s2` 零样本对照仍列入补跑待办。

---

## 四、待他人行动
1. **WorkBuddy**：
   - **优先打包第 13 轮 8B C4 低成本恢复包（预计跑 ~20 分钟）**：仅内置新 [datautils.py](file:///d:/spikellm/SpikeLLM/datautils.py)（`f0786e8472931dfe`）与恢复脚本，不覆盖实验室 `main.py` / `spike_omniquant.py`；先跑 `FP16` 生成 2048 长度 8B C4 缓存（打印 shape + SHA-256），再校验 3 份 `omni_parameters.pth` 完整度并以 `--resume ... --epochs 0 --eval_ppl` 校验 WT2 一致后补出 C4。
   - 顺手核查实验室 `lm_eval` 安装版本（`env_round13_preflight.txt` 显示 `unknown`），为第 14 轮对齐 `lm-eval-v0.4.2` 做好预检。
2. **Codex**：
   - 请核验 [datautils.py](file:///d:/spikellm/SpikeLLM/datautils.py)（`f0786e8472931dfe`，仅改 103/118/167 三处 C4 边界）及本日志对 012 全部 4 项纠偏的落实口径。
3. **用户**：
   - 建议按两步推进：① 立刻让 WorkBuddy 发 20 分钟 8B C4 恢复小包给实验室收口第 13 轮；② 同步授权我把修复完 Codex 3 处口径细节的 **第 14 轮官方 `spcl/QuaRot-GPTQ + Design (B) + Tile-32` 正式脚本归档到 `D:\spikellm\SpikeLLM\`**（独立模块、零侵入现有文件、附 SHA-256 清单并在本地过门禁），实验室收口 8B C4 后即可无缝启动第 14 轮 7B 王炸实验！

---

## 五、风险
1. **实验室 8B `omni_parameters.pth` 完整度未现场核验（低，可前置防呆）**：尽管 `spike_omniquant.py:568` 已在 `eval_ppl` 前完成全部 32 层保存且本地验证位一致，但恢复脚本必须先检查 3 份 `.pth` 的层数（32 层）与 `mask_low_mse` 存在性，并核对恢复后的 WT2 与 `61.5950 / 16.5609 / 14.7334` 一致；若个别不完整再单腿重训。
2. **实验室 `lm_eval` 版本对齐风险（中）**：若实验室 `lm_eval` 版本不是 `v0.4.2`，第 14 轮零样本数字与 SpikeLLM Table 4（`65.61% / 66.76%`）将出现评测口径偏移，需在打包第 14 轮前确认具体版本号。
