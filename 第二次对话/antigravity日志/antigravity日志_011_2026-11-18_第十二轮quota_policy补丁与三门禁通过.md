# Antigravity 日志 011：第十二轮 `--quota_policy` 补丁实装与本地 68M 三门禁全过（2026-11-18）

## 一、用户原话
> 可以，现在就做
>
> 开始做吧
>
> 能不能开始了，究竟怎么回事，重新登录了一遍，还不行吗

---

## 二、我做了什么
1. **实装第十二轮 Stage 2 消融开关 `--quota_policy {adaptive, random, heuristic_down_o}`**：
   - 修改 [main.py](file:///d:/spikellm/SpikeLLM/main.py#L248-L251)：在 `--adaptive_quota` 后新增 `--quota_policy` 参数（`choices=["adaptive", "random", "heuristic_down_o"]`，默认 `"adaptive"`）。
   - 修改 [spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L80-L261)：
     - 抽取共享合法等 MAC 交换计算辅助函数 [`_compute_iso_mac_swap`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L80-L92)，保证所有策略严格满足 `high_MACs <= init_high_macs`；
     - 在 [`allocate_layer_quota`](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L95-L261) 中保留 `"adaptive"` 原有路径逐位不变，新增 `"random"`（确定性种子 `seed*1009 + layer_idx*97 + 13` 随机执行 4 次合法等 MAC 交换）与 `"heuristic_down_o"`（固定先验规则：轮流向 `down_proj` / `o_proj` 倾斜配额，由 `gate_proj / q_proj / up_proj / k_proj / v_proj` 捐赠，最多 6 步，不接触任何评测集）。
   - [spike_linear.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_linear.py) 无需改动，哈希保持 `857be868ad9cc740` 不变。
2. **在本地 RTX 5060 Ti 上跑通 68M（`seed=2, epochs=20`）三项门禁验证**（外部驱动脚本 `verify_r12_quota_policy.py`）。
3. **更新记忆文档**：更新 [.antigravity/MEMORY.md](file:///d:/spikellm/第二次对话/.antigravity/MEMORY.md)。

---

## 三、结论

### 1. 《实验室补丁同步清单》（供 WorkBuddy 打包第十二轮 `patch\`）

| 序号 | 相对 `SpikeLLM/` 路径 | 修改前 `SHA-256前16位` | 修改后 `SHA-256前16位` | 修改后字节数 | 变动说明 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | [main.py](file:///d:/spikellm/SpikeLLM/main.py) | `49485bc7ad4e9e0b` | **`e898ff1da6b92e91`** | `19,293 B` | 新增 `--quota_policy {adaptive,random,heuristic_down_o}` |
| 2 | [spike_driven_quant/spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py) | `2b17e4fcfea74fce` | **`4f7ae925a3c1bd51`** | `24,747 B` | 实装 `random` 与 `heuristic_down_o` 等算力策略 |
| 3 | [spike_driven_quant/spike_linear.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_linear.py) | `857be868ad9cc740` | `857be868ad9cc740`（未变） | `9,487 B` | 无需打包覆盖，仅作基线校验 |

---

### 2. 本地 68M 三门禁实测全部 PASS（`seed=2, addbit=1, low_p=0.9`）

| 门禁项 | 命令行开关 | 实测 `WT2 / C4` | `high_MACs` | 校验结论 |
| :--- | :--- | :---: | :---: | :--- |
| **Gate 0（零回归）** | 无新开关 | `159.2802734375 / 91.741455078125` | `946,176`（基准） | **PASS（逐位一致）**，与修改前 `159.2802734375 / 91.741455078125` 差 `0.0 / 0.0` |
| **Gate 1（Random）** | `--adaptive_quota --quota_policy random` | `157.7954864501953 / 90.99349212646484` | `946,176 -> 946,176 (+0.00%)` | **PASS**，与独立脚本 `rand_s2` 逐位一致 |
| **Gate 2（Heuristic）** | `--adaptive_quota --quota_policy heuristic_down_o` | `157.2209930419922 / 89.42911529541016` | `946,176 -> 946,176 (+0.00%)` | **PASS** |
| **对照：Adaptive（Ours）** | `--adaptive_quota`（默认 `adaptive`） | **`154.4269256591797 / 88.30231475830078`** | `946,176 -> 946,176 (+0.00%)` | 在完全相同的 `946,176` MAC 下显著优于 `Uniform`、`Random` 和固定先验 `Heuristic` |

- **本地 68M 同机严格排序（`seed=2`）**：`Uniform (159.28 / 91.74)` > `Random (157.80 / 90.99)` > `Heuristic down/o (157.22 / 89.43)` > **`Adaptive Ours (154.43 / 88.30)`**。这初步验证了块重构误差自适应搜索比“固定优待 `down/o`”的简单启发式规则更有价值，接下来由第十二轮实验室包在 **7B** 上进行决定性检验。

---

## 四、待他人行动
1. **WorkBuddy**：
   - 请按上方《实验室补丁同步清单》提取本地 [main.py](file:///d:/spikellm/SpikeLLM/main.py)（`e898ff1da6b92e91`）与 [spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py)（`4f7ae925a3c1bd51`），打包第十二轮实验室命令包：
     - 首波必跑：**7B `adaptive` `seed=4`**（补齐 7B 三种子）、**7B `random_quota`（`--adaptive_quota --quota_policy random`, `seed=2`）**、**7B `heuristic_down_o`（`--adaptive_quota --quota_policy heuristic_down_o`, `seed=2`）**、以及 **7B `AQ seed=2` 零样本 5 任务评测**；
     - 提醒：若用补丁机制，请先跑 `68M` 无开关门禁一（预期 `157.788040161133 / 90.831184387207` 逐位一致），并在打补丁后补跑一次 `env_report.py`。
2. **Codex**：
   - 请审阅 [spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py#L222-L246) 中 `heuristic_down_o` 与 `random` 的实现逻辑与预算约束。
3. **用户**：
   - 本轮代码修改与本地全部三项门禁校验已顺利完成（昨晚后台已运行完毕），等待 WorkBuddy 打包第十二轮命令发往实验室即可。

---

## 五、风险
1. **7B `heuristic_down_o` 与 `adaptive` 差距收窄风险（中，待实测）**：在 7B 上绝大多数层（29~30/32）本身就倾向优待 `down_proj / o_proj`，`heuristic_down_o` 在 7B 上的表现可能比在 68M 上更接近 `adaptive`。无论结果如何（自适应明显更好，或与固定规则接近），均为有价值的科学对照。
2. **实验室代码同步（低，可控）**：第十二轮包必须包含新版 [main.py](file:///d:/spikellm/SpikeLLM/main.py) 与 [spike_omniquant.py](file:///d:/spikellm/SpikeLLM/spike_driven_quant/spike_omniquant.py)，不可沿用旧补丁。
