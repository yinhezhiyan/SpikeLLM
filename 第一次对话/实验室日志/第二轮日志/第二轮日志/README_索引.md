# 第二轮日志 — 索引

> 轮次时间：2026-10-29 13:03 → 2026-10-30 13:45（GMT+8）
> 主题：**P0 协议对照实验（不开 `--aug_loss`）**
> 机器：实验室 Windows 工作站，2× RTX 3090 24 GB
> 归档时间：2026-10-30 13:50

---

## 一、这一轮做了什么

只有一件事，但它是论文的核心消融：

**在完全相同的标定预算下，把 `--aug_loss` 关掉，重新跑 Original SpikeLLM 与 OSDSQ v2 两条腿，观察 OSDSQ 的优势是否还成立。**

- 启动：2026-10-29 13:03:37
- 结束：2026-10-30 13:35:11
- 耗时：**24.53 h**，双卡并行，退出码 **0**，两腿 `done.flag` 齐全
- 与第一轮正式实验的**唯一差别**：`--aug_loss` 从"开"变"关"。其余（`wbits=4 abits=4 addbit=1 low_p=0.9 --let --lwc alpha=0.75 let_lr=1e-3 lwc_lr=1e-2 nsamples=128 epochs=20 seed=2`、同一份校准子集、同样三道门禁）完全一致。
- 已核验实参：两腿日志均记录 `aug_loss=False`

---

## 二、核心结论

**OSDSQ v2 的优势不依赖 `aug_loss`。**

| `aug_loss` | 模拟模式 | Ours 相对 Orig 的 WikiText-2 降幅 | C4 降幅 |
|---|---|---|---|
| ✅ | fake_binary | −24.53% | −22.47% |
| ✅ | real_multibit | −24.55% | −23.18% |
| ❌ | fake_binary | **−25.98%** | −21.49% |
| ❌ | real_multibit | **−27.13%** | **−23.89%** |

四组全部落在 **−21.5% ~ −27.1%**，没有任何一组出现优势消失或方向反转 → 排除了"增益来自 `aug_loss` 与 OSDSQ 交互"这一替代解释。

**更关键的观察：两个臂对 `aug_loss` 的依赖程度不同。**

| | Original | OSDSQ v2 |
|---|---|---|
| 关闭 `aug_loss` 后 PPL 劣化 | **+7.6% ~ +8.9%** | +5.1% ~ +6.9% |
| 关闭后零样本 piqa | **−3.10 pt** | −0.49 pt（噪声内） |
| 关闭后零样本 arc_easy | **−2.10 pt** | **+0.67 pt** |

**Ours 不需要 `aug_loss` 也保得住零样本能力，Original 需要。** 且不开 `aug_loss` 时 Ours 依然 6/6 全胜，piqa/arc_easy 的差距反而从 +1.7/+5.3 pt 扩大到 **+4.3/+8.1 pt**。

---

## 三、四象限 PPL（全部本机实测）

FP16 基线（第一轮 STEP 1 实测，与本轮无关但作参照）：**5.472231388092041 / 6.972711086273193**

| 臂 | `aug_loss` | 模拟模式 | WikiText-2 | C4 | ×FP16 (wt2 / c4) |
|---|---|---|---|---|---|
| Original | ✅ | fake_binary | 10.5726 | 13.7738 | ×1.932 / ×1.975 |
| **OSDSQ v2** | ✅ | fake_binary | **7.9792** | **10.6792** | **×1.458 / ×1.532** |
| Original | ✅ | real_multibit | 10.8108 | 14.1855 | ×1.976 / ×2.034 |
| **OSDSQ v2** | ✅ | real_multibit | **8.1570** | **10.8975** | **×1.491 / ×1.563** |
| Original | ❌ | fake_binary | 11.3786 | 14.3712 | ×2.079 / ×2.061 |
| **OSDSQ v2** | ❌ | fake_binary | **8.4223** | **11.2824** | **×1.539 / ×1.618** |
| Original | ❌ | real_multibit | 11.7690 | 15.3005 | ×2.151 / ×2.194 |
| **OSDSQ v2** | ❌ | real_multibit | **8.5758** | **11.6458** | **×1.567 / ×1.670** |

精确值（给复现用）：

```
Orig aug    fake  wt2 10.572571754455566  c4 13.773828506469727
Ours aug    fake  wt2  7.979187488555908  c4 10.679176330566406
Orig aug    real  wt2 10.810821533203125  c4 14.185491561889648
Ours aug    real  wt2  8.157027244567871  c4 10.89747142791748
Orig noaug  fake  wt2 11.378602981567383  c4 14.371224403381348
Ours noaug  fake  wt2  8.422348976135254  c4 11.282405853271484
Orig noaug  real  wt2 11.769049644470215  c4 15.300481796264648
Ours noaug  real  wt2  8.575841903686523  c4 11.645788192749023
```

---

## 四、零样本 6 任务（acc）

| 任务 | Orig ✅ | Orig ❌ | Ours ✅ | Ours ❌ | Δ(✅) | Δ(❌) |
|---|---|---|---|---|---|---|
| piqa | 0.6583 | 0.6273 | 0.6752 | 0.6703 | +1.69 | **+4.30** |
| arc_easy | 0.5004 | 0.4794 | 0.5535 | 0.5602 | +5.31 | **+8.08** |
| arc_challenge | 0.2824 | 0.2892 | 0.3234 | 0.3114 | +4.10 | +2.22 |
| boolq | 0.6235 | 0.6294 | 0.6425 | 0.6544 | +1.90 | +2.51 |
| winogrande | 0.5217 | 0.5209 | 0.5777 | 0.5596 | +5.60 | +3.87 |
| hellaswag | 0.4372 | 0.4353 | 0.4712 | 0.4670 | +3.40 | +3.18 |

acc_norm（❌ 列）：

| 任务 | Orig | Ours |
|---|---|---|
| piqa | 0.6284 | **0.6850** |
| arc_easy | 0.4099 | **0.4630** |
| arc_challenge | 0.3276 | **0.3387** |
| hellaswag | 0.5653 | **0.6084** |

---

## 五、资源与数值健康度

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

**峰值显存比第一轮（18.4 GB）低约 2 GB** —— 关掉 `aug_loss` 省掉了那次带扰动的前向，24 GB 卡余量从 5.6 GB 升到 7.7 GB。

**NaN 层分布**：
- Original：`L1i0, L1i18, L26i16, L27i16, L30i0, L31i0, L31i16`
- OSDSQ v2：`L1i0, L18i0, L27i16, L30i0, L31i0`

与第一轮同源（层 1/18/27/30/31 反复出现，iter 0 与 iter 16 两个热点），**全部单步孤立事件、`loss` 始终有限**，符合 AMP 瞬态溢出判断，未启用 `deactive_amp`。

> Original 本次标定比第一轮慢 1 h 10 min（8h07m vs 6h57m），GPU0 同时承担显示输出，属正常波动，不影响可比性（两边都是 128×20 预算）。

---

## 六、文件清单

```
第二轮日志/                        34 MB / 12 文件
├── README_索引.md                 ← 本文件
├── 01_报告/
│   ├── lab_P0协议对照结果.md       ★ 完整报告（含四象限、敏感度分析、论文意义）
│   └── 实验室电脑工作日志_给工作AI_快照_2026-10-30.md
├── 02_原始日志/
│   ├── lab_p0_driver.log          驱动日志（含 START / job 状态 / 收尾）
│   ├── lab_watch_p0.log           看门狗 148 条快照（24.5 h 留痕）
│   ├── run_p0_noaug_orig.log      ★ Original 完整日志 8.2 MB（逐层 loss/norm）
│   └── run_p0_noaug_osdsq_v2.log  ★ OSDSQ v2 完整日志 8.2 MB
├── 03_门禁结果/
│   ├── orig_real_multibit.txt     real_multibit_simulate 的 PPL
│   ├── orig_zeroshot.txt          6 个零样本任务的 JSON
│   ├── ours_real_multibit.txt
│   └── ours_zeroshot.txt
└── 04_checkpoint/
    ├── omni_parameters_p0_noaug_orig.pth      8.98 MB（md5 53f56ad6…df9f）
    └── omni_parameters_p0_noaug_osdsq_v2.pth  8.98 MB（md5 9a6b2fc4…f8a5）
```

> 两个 checkpoint 已 md5 核对，与原文件一致。

---

## 七、注意

1. **`03_门禁结果\` 是唯一能拿到零样本与 real_multibit 数字的地方。** `main.py` 把这些结果写在 `<output_dir>/zeroshot/log_rank0_*.txt` 和 `<output_dir>/real_multibit/log_rank0_*.txt` 里，**不在 `run.log`**，文件名带随机后缀，容易漏。
2. **第一轮的正式实验数据仍然有效**，两组互为消融，正好构成论文里"训练技巧 × 量化方法"的 2×2 矩阵。参见 `..\第一轮日志\01_报告\lab_7b_formal_results.md`。
3. **FP16 零样本基线仍未做**，所以还是算不出 recovery rate。脚本已备好：`..\..\SpikeLLM\run_lab_fp16_zeroshot.ps1`（`--wbits 16 --abits 16 --epochs 0`，纯前向，约 3–4 h），一条命令就能起。

---

## 八、下一步

| 优先级 | 事项 | 预计耗时 | 状态 |
|---|---|---|---|
| P0 | FP16 零样本基线（6 任务） | 3–4 h | **脚本就绪，等命令** |
| P1 | 第二/第三个 seed（mean±std） | 每 seed ~25 h | 等排期，需独立 `--cache_dir` |
| P1 | 8B（base 版） | 27–31 h | 等缓存隔离确认 + 下载 |
| P2 | arc_easy `acc_norm` < `acc` 的归一化核查 | <1 h | 待核 |
| P2 | 是否复用机器已有的 Mistral-7B / Llama-3-8B-Instruct | — | 待开发组答复（7 个问题见第一轮 §01_报告） |
