# 第一轮日志 — 索引

> 轮次时间：2026-10-27 → 2026-10-29 13:30（GMT+8）
> 机器：实验室 Windows 工作站，2× RTX 3090 24 GB
> 项目根：`C:\Users\admin\Desktop\spikellm\SpikeLLM`
> 归档时间：2026-10-29 13:30

---

## 一、这一轮做了什么（按时间顺序）

| # | 阶段 | 时间 | 结果 |
|---|---|---|---|
| 1 | 环境打通 + 依赖核查 | 10-27 15:26 | Python 3.11.11 / torch 2.6.0+cu126 / transformers 4.44.0，双卡可用，`PREFLIGHT CHECK PASSED` |
| 2 | 下载 LLaMA-2-7B + WikiText2/C4 | 10-27 15:30 → 21:14 | 13,476,876,272 B，0 个 `.incomplete`。中途暴露本机 `os.symlink()` 静默失效，写了 `lab_compat.py` |
| 3 | Smoke 门禁（8 样本 × 1 epoch） | 10-27 21:20 → 10-28 01:05 | 5 次尝试，修掉 5 个 transformers 4.44 兼容问题；两腿 32/32 层标定 + 4 个 resume 门禁全过 |
| 4 | 补 FP16 基线 | 10-27 23:36 | **5.4722 / 6.9727**，与 Llama-2-7B 公认值逐位吻合 → 证明评测链路可信 |
| 5 | V2 部署包覆盖 + 数值回归 | 10-28 10:24 | 回归 26.719755172729492 / 34.46274185180664，与 smoke 逐位一致 |
| 6 | **正式双卡 7B 实验** | 10-28 10:40 → 10-29 11:31 | **24.85 h，退出码 0**，两腿 32/32 层标定完成，3 道门禁全过 |
| 7 | 项目整体迁移 | 10-29 12:42 → 13:03 | 迁到 `Desktop\spikellm\`，270 文件 0 mismatch，回归**逐位一致** |
| 8 | **P0 协议对照启动** | 10-29 13:03:37 | 双卡并行，**仍在运行**，预计 10-30 13:30 完成 |

---

## 二、核心数字（全部本机实测，非文献值）

### PPL

| 配置 | 模拟模式 | WikiText-2 | C4 |
|---|---|---|---|
| FP16 基线 | — | **5.472231388092041** | **6.972711086273193** |
| Original SpikeLLM | fake_binary | 10.5726 | 13.7738 |
| **OSDSQ v2** | fake_binary | **7.9792** | **10.6792** |
| Original SpikeLLM | real_multibit | 10.8108 | 14.1855 |
| **OSDSQ v2** | real_multibit | **8.1570** | **10.8975** |

- 相对 FP16 恶化倍数：Original ×1.93–2.03；**OSDSQ v2 ×1.46–1.56**
- OSDSQ v2 相对 Original 降幅：**22.5%–24.6%**，两种模拟模式几乎同幅度

### 零样本（acc）

| 任务 | Original | OSDSQ v2 | Δ (pt) |
|---|---|---|---|
| piqa | 0.6583 | 0.6752 | +1.69 |
| arc_easy | 0.5004 | 0.5535 | +5.31 |
| arc_challenge | 0.2824 | 0.3234 | +4.10 |
| boolq | 0.6235 | 0.6425 | +1.90 |
| winogrande | 0.5217 | 0.5777 | +5.60 |
| hellaswag | 0.4372 | 0.4712 | +3.40 |

**6 任务 10 指标全部 OSDSQ v2 占优**，幅度远超 stderr（0.005–0.012）。

### 资源

- 32 层标定：Orig 6 h 57 min / Ours 7 h 03 min（约 13.0 min/层）
- 峰值显存：18 379 MB / 18 346 MB（24 GB 卡余量约 5.6 GB）
- 整轮 24.85 h，其中零样本占 14.6–16.9 h
- 硬件：2× RTX 3090 24 GB，驱动 610.88，255.66 GB 内存，Windows 11 Pro

### 数值健康度

| | Original | OSDSQ v2 |
|---|---|---|
| `norm:nan` | 11 次（层 1,18,19,22,24,26,27,29,30,31） | 5 次（层 1,18,30,31） |
| `norm:inf` | 0 | 0 |
| `loss` 非有限 | **0** | **0** |

16 次 NaN **全部为单步孤立事件**（前一个 epoch 有限 → 该步 NaN → 下一步立即恢复），`GradScaler` 跳过该步、权重未污染。按开发组判据允许继续，**未启用 `deactive_amp`，未改动量化算法**。

---

## 三、文件清单与阅读顺序

如果只读两份，读 `01_报告/实验室电脑工作日志_给工作AI.md`（完整同步日志，含代码补丁清单、3 个阻塞项、8 问答复）和 `01_报告/lab_7b_formal_results.md`（正式实验完整结果表）。

```
第一轮日志/                                    20 MB / 35 文件
├── README_索引.md                              ← 本文件
├── 01_报告/
│   ├── 实验室电脑工作日志_给工作AI.md           ★ 主同步日志（累计，668 行）
│   ├── lab_environment_and_smoke_report.md      环境检查 + smoke 报告
│   ├── lab_7b_formal_results.md                 ★ 正式实验完整结果表
│   └── lab_现有大模型可复用性评估.md            ★ 机器上已有的 3 个 LLM 能否直接用（含 7 个待确认问题）
├── 02_环境与下载/
│   ├── lab_check_final.log                      环境检查（PREFLIGHT CHECK PASSED）
│   ├── lab_download_7b.log / _retry1 / _c4fix / _c4fix2   下载 4 次尝试
│   ├── lab_dataset_repair.log / repair2         数据集缓存修复（os.symlink 问题）
│   └── lab_zeroshot_dataset_prefetch.log        零样本数据集预缓存
├── 03_Smoke与门禁/
│   ├── lab_smoke_7b_run2..run5.log              smoke 5 次尝试（run5 成功）
│   ├── lab_gate_orig_piqa.log / lab_gate_ours_piqa.log     零样本门禁
│   ├── lab_gate_ours_real_capture.log           real_multibit 结果副本
│   ├── lab_fp16_baseline_ref.log                FP16 基线 5.4722 / 6.9727
│   └── lab_v2_regression_osdsq.log              V2 部署后的 26.72 / 34.46 回归
├── 04_正式实验/
│   ├── lab_7b_parallel_driver.log               双卡驱动日志（含 STEP1/STEP2 分段）
│   ├── lab_watch.log                            看门狗 147 条快照（24 h 留痕）
│   ├── run_7b_orig_with_let.log                 ★ Original 完整日志 8.2 MB（逐层 loss/norm）
│   └── run_7b_osdsq_v2_with_let.log             ★ OSDSQ v2 完整日志 8.2 MB
├── 05_迁移与当前P0/
│   ├── lab_migrate_regression.log               迁移后回归（7.9792 / 10.6792，逐位一致）
│   ├── migration_manifest_270files.json         迁移前 270 文件清单（路径+大小+mtime）
│   ├── lab_p0_driver.log                        P0 驱动日志（快照）
│   ├── lab_watch_p0.log                         P0 看门狗（快照）
│   ├── run_p0_noaug_orig_快照.log               ★ P0 正在跑，这是 13:30 的快照
│   └── run_p0_noaug_osdsq_v2_快照.log           ★ 同上
├── 06_代码改动留档/
│   ├── run_lab_dual3090_7b_8b.ps1               已加 HF_HOME + ErrorActionPreference 修复
│   ├── run_lab_p0_protocol.ps1                  ★ 新建，P0 协议对照（不带 --aug_loss）
│   ├── lab_watch_progress.ps1                   已参数化 -Arms/-LogName/-Hours
│   └── lab_compat.py                            强制 hub 走复制而非软链
└── 07_checkpoint/
    ├── omni_parameters_7b_orig_with_let.pth       ★ Original 标定产物 8.98 MB（md5 db1066bd…4ccc）
    └── omni_parameters_7b_osdsq_v2_with_let.pth   ★ OSDSQ v2 标定产物 8.98 MB（md5 45c1b5df…f17b）
```

> 两个 checkpoint 已做 md5 校验，与原文件完全一致。它们是 32 层标定的最终产物（LET 的 scale/shift + LWC 的 clip 参数）。

---

## 四、⚠️ 需要注意的几点

1. **`05_` 里的 P0 文件是快照，不是最终结果。** P0 仍在跑（2026-10-29 13:03:37 起，预计 10-30 13:30 完成）。最终结果要到 `SpikeLLM\log\p0_noaug_orig\` 与 `...\p0_noaug_osdsq_v2\` 取，届时会出第二轮的归档。
2. **13 GB 的模型权重不在本目录**（`SpikeLLM\models\Llama-2-7b-hf\`），太大不适合传输；要复现的话直接用项目目录里那份。
3. **`04_` 里的两份 `run.log` 是原始日志**，其中 `real_multibit` 与 `zeroshot` 的结果存在各自子目录的 `log_rank0_*.txt` 里，不在 `run.log` 内 —— 这是 `main.py` 的写法，容易找错。本目录只归档了 `run.log`，需要子目录结果的话我再补。
4. **本文件是副本，不是唯一副本。** 主同步日志的"活文件"仍在 `Desktop\spikellm\实验室电脑工作日志_给工作AI.md`，会持续更新；`01_报告` 里的是 13:30 的快照。

---

## 五、还没做的事（等开发组决定）

1. **FP16 零样本基线**：STEP 1 只评了 FP16 的 PPL，没评零样本，所以现在算不出 recovery rate。补一次约 3–4 h。
2. **多 seed**：目前只有 seed=2 单次，要 mean±std 需再跑 2 个 seed（每轮约 25 h）。注意每 seed 必须独立 `--cache_dir`。
3. **arc_easy 的 `acc_norm`（0.43/0.46）低于 `acc`（0.50/0.55）**，两腿同向，怀疑长度归一化有问题，需要核 `lm_eval/tasks/arc_easy.py`。
4. **8B 未启动**：本机只有 **Instruct** 版（14.97 GiB，完整可用），没有 base 版。另发现"缓存键冲突"其实是**命名造成的**——目录名叫 `Meta-Llama-3-8B` 时 `model_family="Meta"`，与 7B 的 `"Llama"` 不冲突，可能不用改代码就能解掉。
5. **是否复用机器上已有的另外两个 LLM**（Mistral-7B-Instruct-v0.3 结构同构可直接插、Qwen2.5-7B 因 qkv bias 不行）—— 详见 `01_报告/lab_现有大模型可复用性评估.md`，里面有 7 个待确认问题。
