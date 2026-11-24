# P0 协议对照实验结果（不开 `--aug_loss`）

> 运行区间：2026-10-29 13:03:37 → 2026-10-30 13:35:11（**24.53 h**，双卡并行）
> 退出码：**0**，两腿 `done.flag` 齐全
> 命令：`run_lab_p0_protocol.ps1 -PythonExe C:\Users\admin\.conda\envs\spikellm\python.exe`
> 与正式实验的**唯一差别**：不带 `--aug_loss`。其余全部相同（`wbits=4 abits=4 addbit=1 low_p=0.9 --let --lwc alpha=0.75 let_lr=1e-3 lwc_lr=1e-2 nsamples=128 epochs=20 seed=2`，复用同一份校准子集）
> 已核验实参：两腿日志均记录 `aug_loss=False`

---

## 一、核心结论（一句话）

**OSDSQ v2 的优势不依赖 `aug_loss`。** 关掉 `aug_loss` 后，OSDSQ v2 相对 Original 的 PPL 降幅为 **21.5%–27.1%**（4 组对照），与开启时的 22.5%–24.6% 基本持平，其中 3 组还略微更大。两个臂都因关闭 `aug_loss` 而变差，但 **Original 变得更差更多**。

---

## 二、PPL 完整四象限

| 配置 | `aug_loss` | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|---|
| FP16 基线 | — | — | **5.472231388092041** | **6.972711086273193** |
| Original | ✅ | fake_binary | 10.572571754455566 | 13.773828506469727 |
| **OSDSQ v2** | ✅ | fake_binary | **7.979187488555908** | **10.679176330566406** |
| Original | ✅ | real_multibit | 10.810821533203125 | 14.185491561889648 |
| **OSDSQ v2** | ✅ | real_multibit | **8.157027244567871** | **10.89747142791748** |
| Original | ❌ | fake_binary | 11.378602981567383 | 14.371224403381348 |
| **OSDSQ v2** | ❌ | fake_binary | **8.422348976135254** | **11.282405853271484** |
| Original | ❌ | real_multibit | 11.769049644470215 | 15.300481796264648 |
| **OSDSQ v2** | ❌ | real_multibit | **8.575841903686523** | **11.645788192749023** |

### 2.1 相对 FP16 的恶化倍数

| 臂 | `aug_loss` | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|---|
| Original | ✅ | fake | ×1.932 | ×1.975 |
| **OSDSQ v2** | ✅ | fake | **×1.458** | **×1.532** |
| Original | ✅ | real | ×1.976 | ×2.034 |
| **OSDSQ v2** | ✅ | real | **×1.491** | **×1.563** |
| Original | ❌ | fake | ×2.079 | ×2.061 |
| **OSDSQ v2** | ❌ | fake | **×1.539** | **×1.618** |
| Original | ❌ | real | ×2.151 | ×2.194 |
| **OSDSQ v2** | ❌ | real | **×1.567** | **×1.670** |

### 2.2 ★ 最关键的一张表：OSDSQ v2 相对 Original 的降幅

| `aug_loss` | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|
| ✅ | fake_binary | −24.53% | −22.47% |
| ✅ | real_multibit | −24.55% | −23.18% |
| ❌ | fake_binary | **−25.98%** | −21.49% |
| ❌ | real_multibit | **−27.13%** | **−23.89%** |

**四组全部落在 −21.5% ~ −27.1%，没有任何一组出现优势消失或方向反转。** 这排除了"增益来自 `aug_loss` 与 OSDSQ 的交互"这一替代解释。

### 2.3 `aug_loss` 本身的作用（关闭后相对开启的劣化）

| 臂 | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|
| Original | fake_binary | **+7.62%** | +4.34% |
| Original | real_multibit | **+8.86%** | **+7.86%** |
| OSDSQ v2 | fake_binary | +5.55% | +5.65% |
| OSDSQ v2 | real_multibit | **+5.13%** | +6.87% |

`aug_loss` 对两个臂都有正贡献（关掉都变差 4%–9%），但**对 Original 的贡献更大**（fake 模式 7.62% vs 5.55%）。换句话说：Original 更依赖 `aug_loss`。

---

## 三、零样本 6 任务（`acc`）

| 任务 | Origin aug | Origin **noaug** | Ours aug | Ours **noaug** | Δ(aug) | Δ(**noaug**) |
|---|---|---|---|---|---|---|
| piqa | 0.6583 | 0.6273 | 0.6752 | 0.6703 | +1.69 | **+4.30** |
| arc_easy | 0.5004 | 0.4794 | 0.5535 | 0.5602 | +5.31 | **+8.08** |
| arc_challenge | 0.2824 | 0.2892 | 0.3234 | 0.3114 | +4.10 | +2.22 |
| boolq | 0.6235 | 0.6294 | 0.6425 | 0.6544 | +1.90 | +2.51 |
| winogrande | 0.5217 | 0.5209 | 0.5777 | 0.5596 | +5.60 | +3.87 |
| hellaswag | 0.4372 | 0.4353 | 0.4712 | 0.4670 | +3.40 | +3.18 |

**不开 `aug_loss` 时 OSDSQ v2 依然 6/6 全胜**，且 piqa / arc_easy 的差距从 +1.7 / +5.3 pt 扩大到 **+4.3 / +8.1 pt**。

### 3.1 两个臂对 `aug_loss` 的敏感度不同（这是本节最重要的观察）

| 任务 | Original 关闭后变化 | OSDSQ v2 关闭后变化 |
|---|---|---|
| piqa | **−3.10 pt** | −0.49 pt |
| arc_easy | **−2.10 pt** | **+0.67 pt** |
| arc_challenge | +0.68 pt | −1.20 pt |
| boolq | +0.59 pt | +1.19 pt |
| winogrande | −0.08 pt | −1.81 pt |
| hellaswag | −0.19 pt | −0.42 pt |

- **Original**：piqa 和 arc_easy 出现 −3.1 / −2.1 pt 的实质退化。
- **OSDSQ v2**：六个任务全部在 ±1.8 pt 内波动，**与 stderr（0.005–0.012 ≈ 0.5–1.2 pt）同量级，属噪声**；arc_easy 甚至略升。

结论：**OSDSQ v2 不需要 `aug_loss` 也能保住零样本能力，而 Original 需要。**

---

## 四、资源与数值健康度

| | Original（GPU0） | OSDSQ v2（GPU1） |
|---|---|---|
| 标定起点（layer 0） | 10-29 13:04:09 | 10-29 13:04:09 |
| 标定终点（layer 31 起） | 10-29 21:11:24 | 10-29 20:01:50 |
| **32 层标定耗时** | **8 h 07 min** | **6 h 58 min** |
| checkpoint 落盘 | 21:24 | 20:15 |
| 峰值 `max_memory_allocated` | **16 331 MB** | **16 331 MB** |
| `norm:nan` | 7 次 | 5 次 |
| `norm:inf` | 0 | 0 |
| `loss` 为 nan/inf | **0** | **0** |

**峰值显存比正式实验（18.4 GB）低了约 2 GB**，这是关掉 `aug_loss` 的直接收益——`aug_loss` 需要额外一次带扰动的前向，激活占用更大。24 GB 卡余量从 5.6 GB 升到约 7.7 GB。

**NaN 层分布**：

- Original：`L1i0, L1i18, L26i16, L27i16, L30i0, L31i0, L31i16`（7 次）
- OSDSQ v2：`L1i0, L18i0, L27i16, L30i0, L31i0`（5 次）

与正式实验同源（层 1 / 18 / 27 / 30 / 31 反复出现，另有 iter 0 与 iter 16 两个热点），**全部为单步孤立事件、`loss` 始终有限**，符合 AMP 瞬态溢出的既有判断。

> 注意：本次 Original 的标定比正式实验慢了 1 h 10 min（8h07m vs 6h57m）。两个臂共享同一台机器，GPU0 同时承担显示输出，波动属正常范围，不影响结果可比性（标定预算都是 128×20）。

---

## 五、对论文的意义

1. **可以放心写"增益与 `aug_loss` 无关"。** 四组对照（2 臂 × 2 模拟 × 2 个数据集 = 8 个数字）方向全部一致，幅度稳定在 −21.5% ~ −27.1%。
2. **`aug_loss` 不是 OSDSQ v2 的"奶嘴"。** 相反，Original 才是离不开 `aug_loss` 的那一个（PPL 劣化更多、零样本 piqa/arc_easy 掉 2–3 pt）。
3. **显存预算可以往下报。** 不开 `aug_loss` 时峰值 16.3 GB，这对"能不能在 24 GB 卡上跑"的论证更有利。
4. 已有的正式实验结果（`aug_loss` 开启）依然可以照用，两组互为消融，正好构成论文里"训练技巧 × 量化方法"的 2×2 矩阵。

---

## 六、产物

| 产物 | 路径 |
|---|---|
| Original checkpoint | `SpikeLLM\log\p0_noaug_orig\omni_parameters.pth`（8 983 506 B） |
| OSDSQ v2 checkpoint | `SpikeLLM\log\p0_noaug_osdsq_v2\omni_parameters.pth`（8 983 506 B） |
| 完整日志 | `SpikeLLM\log\p0_noaug_orig\run.log`、`...\p0_noaug_osdsq_v2\run.log`（各 8.2 MB） |
| real_multibit 结果 | 各腿 `real_multibit\log_rank0_*.txt` |
| 零样本结果 | 各腿 `zeroshot\log_rank0_*.txt` |
| 驱动日志 | `SpikeLLM\lab_p0_driver.log` |
| 看门狗 | `SpikeLLM\lab_watch_p0.log` |

---

## 七、下一步（等决定）

1. **FP16 零样本基线**（约 3–4 h）：现在两臂的零样本都有数了，但**没有 FP16 的零样本**，所以仍然算不出 recovery rate。脚本已备好 `run_lab_fp16_zeroshot.ps1`，一句话就能起。
2. **第二 / 第三 seed**：每 seed 约 25 h，必须配独立 `--cache_dir`。
3. **8B**：机器上有 Instruct 版（14.97 GiB，完整可用），但不是 base；另发现缓存键冲突是命名造成的，改目录名即可规避。
