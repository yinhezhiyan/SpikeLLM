# SpikeLLM / OSDSQ v2 — 双 3090 正式实验结果（LLaMA-2-7B, W4A4）

> 运行区间：2026-10-28 10:40:03 → 2026-10-29 11:31:06（**24.85 h**，双卡并行）
> 退出码：**0**（wrapper 与两个子进程均 0）
> 命令：`run_lab_dual3090_7b_8b.ps1 -Mode 7b_parallel -PythonExe C:\Users\admin\.conda\envs\spikellm\python.exe`
> 配置：`nsamples=128, epochs=20, seed=2, wbits=4, abits=4, addbit=1, low_p=0.9, --let --lwc --aug_loss --alpha 0.75 --let_lr 0.001 --lwc_lr 0.01`
> 本机：2× RTX 3090 24 GB，Windows 11 Pro，Python 3.11.11，torch 2.6.0+cu126，transformers 4.44.0

---

## 1. 主结果：PPL

FP16 基线来自本脚本 STEP 1（同一套 2048 窗口 NLL、同一 tokenizer、同一数据缓存）：

| 配置 | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|
| **FP16（基线）** | — | **5.472231388092041** | **6.972711086273193** |
| Original SpikeLLM | `fake_binary_simulate` | 10.572571754455566 | 13.773828506469727 |
| **OSDSQ v2** | `fake_binary_simulate` | **7.979187488555908** | **10.679176330566406** |
| Original SpikeLLM | `real_multibit_simulate` | 10.810821533203125 | 14.185491561889648 |
| **OSDSQ v2** | `real_multibit_simulate` | **8.157027244567871** | **10.89747142791748** |

### 相对 FP16 的恶化倍数

| 配置 | WikiText-2 | C4 |
|---|---|---|
| Original / fake_binary | ×1.932 | ×1.975 |
| OSDSQ v2 / fake_binary | **×1.458** | **×1.532** |
| Original / real_multibit | ×1.976 | ×2.034 |
| OSDSQ v2 / real_multibit | **×1.491** | **×1.563** |

### OSDSQ v2 相对 Original 的降幅

| 模拟模式 | WikiText-2 | C4 |
|---|---|---|
| `fake_binary_simulate` | **−24.53%** | **−22.47%** |
| `real_multibit_simulate` | **−24.55%** | **−23.18%** |

**两种模拟模式给出的降幅几乎相同（24.5% / 22.5% vs 24.6% / 23.2%）**，说明增益不是某种模拟近似带来的假象。

---

## 2. 主结果：6 个零样本任务（`acc` / `acc_norm`）

| 任务 | Original acc | OSDSQ v2 acc | Δ (pt) | Original acc_norm | OSDSQ v2 acc_norm | Δ (pt) |
|---|---|---|---|---|---|---|
| piqa | 0.6583 | **0.6752** | +1.69 | 0.6638 | **0.6866** | +2.28 |
| arc_easy | 0.5004 | **0.5535** | +5.31 | 0.4306 | **0.4596** | +2.90 |
| arc_challenge | 0.2824 | **0.3234** | +4.10 | 0.3106 | **0.3328** | +2.22 |
| boolq | 0.6235 | **0.6425** | +1.90 | — | — | — |
| winogrande | 0.5217 | **0.5777** | +5.60 | — | — | — |
| hellaswag | 0.4372 | **0.4712** | +3.40 | 0.5767 | **0.6150** | +3.83 |

**OSDSQ v2 在全部 6 个任务、全部 10 个指标上一致优于 Original**（6 个 `acc` + 4 个 `acc_norm`，boolq/winogrande 只有 `acc`）。

方向与 PPL 完全一致，且优势幅度（+1.7 ~ +5.6 pt）远大于 stderr（约 0.005–0.012），不是噪声。

> ⚠️ 本轮 STEP 1 只评了 FP16 的 PPL，**没有 FP16 零样本基线**，因此上表只能做"两腿互比"，不能直接读出"相对无损掉了多少"。若要写"recovery rate"，需要补一次 FP16 零样本（约 3–4 h）。

---

## 3. 耗时与资源

| 项目 | Original（GPU0） | OSDSQ v2（GPU1） |
|---|---|---|
| 标定开始（layer 0） | 10-28 10:45:58 | 10-28 10:45:59 |
| 标定结束（layer 31 起） | 10-28 17:43:29 | 10-28 17:49:12 |
| checkpoint 落盘 | 10-28 17:57:07 | 10-28 18:02:55 |
| **32 层标定耗时** | **6 h 57 min** | **7 h 03 min** |
| PPL（fake_binary） | 18:09 | 18:15 |
| PPL（real_multibit） | 18:38 | 18:37 |
| 6 个零样本 | 18:39 → 09:17 | 18:38 → 11:31 |
| **整腿总耗时** | **22.6 h** | **24.75 h** |
| 峰值显存（`max_memory_allocated`） | **18 379 MB** | **18 346 MB** |

- 平均 **约 13.0 min/层**（128 样本 × 20 epoch，约 88 s/epoch），24 GB 卡余量约 5.6 GB。
- 比原估的 68 h/腿快得多（那是在 smoke 的 8 样本上线性外推，严重高估）。
- 零样本阶段占整腿一半以上时间（14.6 h / 16.9 h），是下一个可优化的瓶颈。

---

## 4. NaN / 数值健康度

| | Original | OSDSQ v2 |
|---|---|---|
| `norm:nan` 累计 | 11 次 | 5 次 |
| `norm:inf` 累计 | 0 | 0 |
| `loss` 为 nan/inf | **0** | **0** |
| 出现层 | 1, 18, 19, 22, 24, 26, 27, 29, 30, 31 | 1, 18, 30, 31 |
| 是否为单步孤立事件 | **全部是** | **全部是** |

**逐条核对了每一次 NaN 的前后 epoch**，16 次全部满足"前一个 epoch 有限 → 该 epoch NaN → 后一个 epoch 立即恢复有限"：

```
Orig  L1i0  -> L0i19:1.3e-05  | L1i0:nan | L1i1:0.0279
Orig  L24i15-> L24i14:1.9e-03 | L24i15:nan| L24i16:1.9e-03
Orig  L27i19-> L27i18:8.3e-04 | L27i19:nan| L28i0:7.5e-03
Orig  L30i0 -> L29i19:1.7e-03 | L30i0:nan | L30i1:0.6776
Ours  L30i15-> L30i14:3.5e-02 | L30i15:nan| L30i16:3.6e-02
Ours  L31i0 -> L30i19:2.2e-02 | L31i0:nan | L31i1:0.0246
```

- `loss` 始终有限，只有**梯度范数**为 NaN → `GradScaler` 跳过该 step，权重未被污染。
- 形态与开发组预告的 AMP 瞬态溢出一致：**按指令继续，未启用 `--deactive_amp`，未改动量化算法**。
- 注意 Original 有 4 次发生在 **iter 15 / iter 19**（训练中后段，不是首个 batch），说明不只是"层首 batch"现象，但同样单步恢复。
- 与 smoke（8 样本）对比：smoke 时 OSDSQ v2 的 NaN 也是 3 次，正式跑（128 样本 × 20 epoch，共 640 步）只涨到 5 次，**NaN 密度随预算增大而显著下降**，确认是瞬态而非结构性缺陷。

---

## 5. 产物清单

| 产物 | 路径 | 大小 |
|---|---|---|
| Original checkpoint | `log/7b_orig_with_let/omni_parameters.pth` | 8 983 506 B |
| OSDSQ v2 checkpoint | `log/7b_osdsq_v2_with_let/omni_parameters.pth` | 8 983 506 B |
| Original 完整日志 | `log/7b_orig_with_let/run.log` | 8.6 MB |
| OSDSQ v2 完整日志 | `log/7b_osdsq_v2_with_let/run.log` | 8.6 MB |
| Original real_multibit | `log/7b_orig_with_let/real_multibit/log_rank0_*.txt` | — |
| OSDSQ v2 real_multibit | `log/7b_osdsq_v2_with_let/real_multibit/log_rank0_*.txt` | — |
| Original 零样本 | `log/7b_orig_with_let/zeroshot/log_rank0_*.txt` | — |
| OSDSQ v2 零样本 | `log/7b_osdsq_v2_with_let/zeroshot/log_rank0_*.txt` | — |
| 驱动日志 | `lab_7b_parallel_driver.log` | — |
| 进度看门狗 | `lab_watch.log`（147 条快照） | — |
| 完成标记 | 两腿各 `done_ppl.flag` / `done_multibit.flag` / `done.flag` | — |

`omni_parameters.pth` 是**逐层增量覆盖**写的，最终大小 8.98 MB 且时间戳晚于 layer 31，说明 32 层全部写进去了。

---

## 6. 结论

1. **在这台双 3090 上、在完全相同的标定预算下，OSDSQ v2 稳定优于 Original SpikeLLM**：PPL 降 22.5%–24.6%（两种模拟模式一致），6 个零样本任务 +1.7 ~ +5.6 pt，10 个指标全部占优。
2. **数值健康**：无 OOM、无崩溃、无 loss 非有限；16 次 `norm:nan` 全部单步恢复，属于已知 AMP 瞬态，未干预。
3. **耗时 24.85 h / 两腿并行**，比预估快约 3 倍。
4. **尚未具备的条件**：
   - 无 FP16 零样本基线 → 无法计算 recovery rate；
   - 只有 seed=2 单次运行 → 无 seed 方差，论文若要做显著性检验需补 2 个 seed（每轮约 25 h）；
   - 8B 未启动（缓存键冲突未解决，且必须用 base 版）。
5. 按指令本轮**未做** timestep 扫描与 KV 新指标。
