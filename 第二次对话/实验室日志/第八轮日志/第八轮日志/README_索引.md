# 第八轮日志索引 · addbit 1 → 3（2026-11-06 20:52 → 11-07 04:44）

## 一、本轮唯一要回答的问题

前七轮全部跑在 `--addbit 1`（显著通道只多 1 bit）。审稿人会问："为什么不直接把 addbit 调大？"
本轮把 addbit 从 1 提到 3，其余参数与第七轮逐字相同，看 +3 bit 能不能把 Original 救回来。

**结论（按补丁里写死的判据，未做二次解释）**

| 量 | 值 | 判据 |
|---|---|---|
| A = 7B Orig@a3 wt2 | **10.1386** | `A >= 9.0` → 原叙事成立，addbit 救不回来 |
| B = 7B Ours@a3 wt2 | **7.8978** | B 比 A 小 22.10%（>5%）→ 方法在高 T 下仍有效 |

## 二、核心数字

### 7B 主实验（32 层，双卡并行，21:25 → 04:44，7 h 19 m，640 iter/腿，40 s/iter）

| 腿 | wt2@a3 | c4@a3 | wt2@T2 | c4@T2 | wt2 Δ | c4 Δ |
|---|---|---|---|---|---|---|
| Orig | 10.1386 | 12.8944 | 10.5726 | 13.7738 | −4.10% | −6.38% |
| Ours | **7.8978** | **10.5766** | 7.9792 | 10.6792 | −1.02% | −0.96% |

相对降幅（Ours 相对 Orig，同 addbit 配对）：T=2 是 wt2 24.53% / c4 22.47%；a3 是 wt2 **22.10%** / c4 17.98%。
FP16 参照 5.4722 / 6.9727（同机同版，本轮未变）。

### 68M 门禁（2 层，串行 GPU0，约 30 min）

| 腿 | wt2 | c4 | T=2 参照 |
|---|---|---|---|
| Orig@a3 | 176.0957 | 99.8575 | 172.29 / 175.14（两个种子的 wt2） |
| Ours@a3 | **153.4056** | **88.9996** | 156.84 / 90.53 |

68M 上 Orig 反而比 T=2 两个种子都高（+2.21%），Ours 降 2.19% / 1.69%。68M 噪声大，只作方向判断。

### 稳定性
7B 两条腿 `norm:nan` 分别 17 / 10 次，均为 AMP 瞬态（第七轮同量级 3~12），最终 PPL 正常出数；无 OOM、无崩溃、无断点续跑。

## 三、两条实证旁注（不是判据，回传后三方核对）

1. **addbit=3 并没有复现 T=8。** 补丁说明把两者当成"同一预算的两种记法"，但 Orig 只从 10.5726 走到 10.1386（−4.10%），离 T=8 文献值 5.86 很远。本轮不能证明 addbit 与 T 等价。
2. **`--addbit 3` 只加宽了线性层。** `spike_matmul.py:33` 给 KV/PV 高显著量化器传 `Refine=(addbit>0)`，而 `quantizer.py:47-48` 里 `if Refine: addbit = 1` → K/V 恒 5 bit。a1 与 a3 仍是同口径，A/B 判据不受影响；但要声称"整条高显著预算 +3 bit"，需改 Refine 口径重跑。

## 四、执行记录

- 补丁：`lab_round8_addbit_sweep_2026-11-04_v2.zip` → 单文件 `lab_round8_runner.ps1`（18031 B，md5 `153c76a7e668c609e35c88c5b2dc07c0`，原封未改，见 `06_环境与预检/`）。
- **本轮零代码改动**：8 个关键文件哈希与第七轮逐字相同 → 可与第七轮并列进同一张表。哈希见 `06_环境与预检/env_round8_preflight.txt`。
- preflight 6/6 通过（python 3.11.11 / torch 2.6.0+cu126 / transformers 4.44.2；7B 与 68M 的 act_stats 齐；磁盘 112.2 GB）。
- `-Mode run` 一次跑完，exit 0；四条腿 `done_ppl.flag` + `done.flag` 全齐；`lab_round8_result.txt` 8 个数字行无缺。
- 看门狗 `lab_watch_round8.log` 每 10 min 一条，594 行，末行 `===== watchdog stopped: all done =====`，全程无 FATAL 误判。

## 五、文件清单（20 个文件 / 19 MB）

```
01_结果原件/lab_round8_result.txt          ← 原样回传件（脚本自生成，未改一字）
01_结果原件/第八轮回传校验.txt              ← 归档 log_rank0 重取 PPL 与 result.txt 逐位一致
02_原始日志/r8_{68m_orig_a3,68m_ours_a3,7b_orig_a3,7b_ours_a3}_run.log
02_原始日志/log_rank0_*.txt ×4
02_原始日志/lab_watch_round8.log            ← 看门狗 594 行终版快照
02_原始日志/lab_round8_console.log
03_门禁结果/r8_68m_gate_对照.txt            ← 68M 门禁 + 7B 差值表（脚本计算，未手填）
04_checkpoint/*_omni_parameters.pth ×4      ← 7B 两条腿各 8.6 MB，68M 各 136 KB
05_回传包/第八轮回传包_2026-11-07.zip        ← 补丁要求的回传包（result + log_rank0 + env）
06_环境与预检/env_round8_preflight.txt
06_环境与预检/lab_round8_runner.ps1
```

checkpoint md5：`7b_orig ea209362…`、`7b_ours e261a0f5…`、`68m_orig 036626e5…`、`68m_ours 4725f88d…`。

## 六、需要对方决定的事

1. **A = 10.14 是否足以关掉"addbit 是廉价替代"这条质疑**：按写死的判据是"成立"，但 Orig@a3 只降了 4%，若审稿人要求等能量前沿，是否需要补 a2（+2 bit）凑三点成曲线？
2. **addbit 与 T 的关系要不要单独验证**：5.86 那条路径靠的是真实脉冲 T=8（multibit 分支），本轮的 fake_binary addbit 不是同一旋钮。若要报价前沿，得跑 `real_multibit_simulate` 且 `steps=8`。
3. **KV 的 `Refine` 口径**：现在 K/V 高显著恒 5 bit，是否改成随 addbit 走（改代码 + 重跑 2 条腿）？
4. 遗留：ACE / 发放率脚本仍未收到；8B 冒烟未排期；68M mean 基线三值（156.84 / 156.09 / 157.79）仍需定口径。
