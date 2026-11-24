# 第四轮日志 — 索引

> 轮次时间：2026-10-31 02:50 → 10:45（GMT+8），双卡并行
> 主题：**两个 LOO 消融 —— B 和 C 到底有没有贡献**
> 归档时间：2026-10-31 10:55

---

## 一、这一轮做了什么

同时跑掉 `--ablate_no_kv_shield`（去掉 B）和 `--ablate_orig_saliency`（去掉 C），加上第三轮的 `--ablate_orig_loss`（去掉 D），四个组件的消融就凑齐了三个。

| GPU | Tag | 参数 | 耗时 | 退出码 |
|---|---|---|---|---|
| 0 | `loo_minus_kv_shield` | `--ablate_no_kv_shield` | **7.89 h** | 0 |
| 1 | `loo_minus_saliency` | `--ablate_orig_saliency` | **7.61 h** | 0 |

实参已核：两腿 `aug_loss=True`，各自只开一个 ablate。

---

## 二、结论（第三条最重要）

1. **D 是唯一稳健成立的组件**：去掉后 PPL 变差 **+2.0% ~ +4.5%**（0.17–0.45 PPL）。
2. **B 和 C 的效应量只有约 0.1 PPL，且方向不稳定** —— B 在 `real_multibit` 下去掉反而**更好**（−0.54%）。
3. **A（线性层均值锚定屏蔽）根本没有消融开关**，无法 LOO。这是代码缺口。

### LOO 总表

完整配置（A+B+C+D）：fake **7.9792 / 10.6792**，real **8.1570 / 10.8975**

| 去掉 | 开关 | fake wt2 | Δ | fake c4 | Δ | real wt2 | Δ | real c4 | Δ |
|---|---|---|---|---|---|---|---|---|---|
| **D** | `--ablate_orig_loss` | 8.2403 | **+0.261** | 11.1289 | **+0.450** | 8.3221 | **+0.165** | 11.3830 | **+0.486** |
| **B** | `--ablate_no_kv_shield` | 8.1102 | +0.131 | 10.6918 | +0.013 | 8.1131 | **−0.044** | 10.9275 | +0.030 |
| **C** | `--ablate_orig_saliency` | 8.0901 | +0.111 | 10.7502 | +0.071 | 8.1753 | +0.018 | 10.9955 | +0.098 |

（Δ 单位为 PPL，正值 = 去掉后变差 = 该组件有贡献）

---

## 三、⚠️ 怎么读：不是"B/C 没用"，是"分不清"

四个配置全部只跑了 **seed=2 单次**，没有重复，因此**无法判断 0.1 PPL 是真实效应还是运行噪声**。AMP + 逐层标定本身就有这个量级的波动。

**正确的表述**：B/C 的效应量**低于当前测量分辨率**，需要多 seed 才能判定。
**错误的表述**：B/C 没有贡献。（不能这么写，会误删真组件）

---

## 四、代码层面的一个重要缺口：A 没有开关

逐层读量化路径后的开关清单：

| 组件 | 代码位置 | 开关 |
|---|---|---|
| **A** 线性层均值锚定屏蔽 | `spike_linear.py:99` `_quant_low()` | ❌ **无**（只能整个走 `--orig_spikellm`） |
| **B** 注意力 KV 均值锚定屏蔽 | `spike_matmul.py:102` `_quant_x2_low()` | ✅ `--ablate_no_kv_shield` |
| **C** 步长加权显著性 | `spike_linear.py:87` / `spike_matmul.py:74` `get_channel_mask()` | ✅ `--ablate_orig_saliency` |
| **D** 加权第二目标+余弦 | `spike_omniquant.py:311-318` | ✅ `--ablate_orig_loss`（需 `--aug_loss`） |

**额外发现**：B 只在 `is_p == True` 分支生效（`spike_matmul.py` 里 `if self.is_p:` 有守卫），`is_p == False` 时即使不开 ablate 也走原始路径 → **B 的实际作用面本来就比名字暗示的小**，这可能是它效应量小的原因。

**建议开发组加 `--ablate_no_linear_shield`**，否则 A 这个"主组件"在消融表里是空白，审稿人必问。

---

## 五、资源

| | KV（GPU0） | 显著性（GPU1） |
|---|---|---|
| 整轮 | 7.89 h | 7.61 h |
| 峰值显存 | 18 346 MB | 18 346 MB |
| `norm:nan` | 6 | 6 |
| `norm:inf` / loss 非有限 | 0 / 0 | 0 / 0 |

对照：完整 D 是 5 次 NaN、等权第二目标 9 次。三个 LOO 都约 6 次，全部单步恢复。

---

## 六、文件清单

```
第四轮日志/                18 MB / 10 文件
├── README_索引.md
├── 01_报告/
│   ├── lab_LOO消融结果.md                        ★ 完整报告（含代码级开关清单、论文建议）
│   └── 实验室电脑工作日志_给工作AI_快照_2026-10-31.md
├── 02_原始日志/
│   ├── lab_loo_kv_driver.log
│   ├── lab_loo_saliency_driver.log
│   ├── run_loo_minus_kv_shield.log
│   └── run_loo_minus_saliency.log
├── 03_门禁结果/
│   ├── loo_minus_kv_shield_real_multibit.txt
│   └── loo_minus_saliency_real_multibit.txt
└── 04_checkpoint/
    ├── omni_parameters_loo_minus_kv_shield.pth    md5 231f93a3…c69f
    └── omni_parameters_loo_minus_saliency.pth     md5 fb8dac23…8faf
```

---

## 七、对论文骨架的影响（需要开发组决策）

原计划故事是 **A + B + C + D**。当前证据下：

| 组件 | 可否作为创新点 | 建议 |
|---|---|---|
| A | 无法验证 | 先加开关再跑 LOO，否则不能声称 |
| B | 证据不足 | 不宜作独立贡献，可并入 A 或作为实现细节 |
| C | 证据不足 | 同上 |
| D | 成立 | 可写，但需说明边际收益只有 2%–4% |

**替代叙事**（即使 B/C 不显著也成立）：把故事从"四个组件各有贡献"改成"**一个更强的基座（A）+ 一个必要的训练目标（D）**，A 是主要来源" —— 相对 Original 的差距（PPL −22~−25%、零样本 +3~8 pt）依然巨大。

**但先别删组件**：B/C 的 0.03–0.13 PPL 正好落在"单 seed 分不清"的区间。建议先对"完整配置"与"−B"各跑 seed=3 拿到重复间标准差，再决定去留。

---

## 八、下一步

| 优先级 | 事项 | 状态 |
|---|---|---|
| P0 | 加 `--ablate_no_linear_shield`（开发组） | 等 |
| P0 | seed=3（至少「完整」与「−B」两点）拿标准差 | 等排期 |
| P1 | ACE / 发放率脚本 | 等开发组给口径 |
| P1 | 8B | 等 |
